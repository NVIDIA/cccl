// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Assemble private sorting entries, manage temporary storage, and launch partitioning followed by sorting.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_sort_group_merge.cuh>
#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/segmented_sort_capacity.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort_capacity.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Adapt each member of the combined policy into one implementation and ordered sorting entry.
template <typename PolicyGetter>
struct group_merge_entries
{
  // Each getter exposes just one launch's settings to group_merge_implementation.
  struct warp_policy_getter
  {
    _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const
    {
      return PolicyGetter{}().warp;
    }
  };
  struct small_policy_getter
  {
    _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const
    {
      return PolicyGetter{}().small_block;
    }
  };
  struct large_policy_getter
  {
    _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const
    {
      return PolicyGetter{}().large;
    }
  };
  using type = sort_entries<sort_entry<group_merge_implementation<warp_policy_getter>>,
                            sort_entry<group_merge_implementation<small_policy_getter>>,
                            sort_entry<group_merge_implementation<large_policy_getter>>>;
};

// Query or use temporary storage, then count, prefix, and scatter IDs into capacity bins.
// These pointers alias caller-provided global storage. It must remain valid until
// every operation queued on the stream has completed, as with other CUB algorithms.
// Empty ranges are skipped; singletons use the same sorting path as other ranges.
template <typename Capacities,
          typename BeginIterator,
          typename EndIterator,
          typename Launcher = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION cudaError_t dispatch_partition(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  unsigned num_segments,
  partition_result& result,
  cudaStream_t stream = 0,
  Launcher launcher   = {})
{
  constexpr int bins = Capacities::size + 1;
  // One allocation holds counts, offsets (with an end sentinel), and cursors; the other holds IDs.
  const size_t sizes[] = {sizeof(unsigned) * (3 * bins + 1), sizeof(unsigned) * size_t{num_segments}};
  void* allocations[2]{};
  if (const auto error = CubDebug(alias_temporaries(temp_storage, temp_storage_bytes, allocations, sizes)))
  {
    return error;
  }
  if (!temp_storage)
  {
    return cudaSuccess;
  }
  auto* counts  = static_cast<unsigned*>(allocations[0]);
  auto* offsets = counts + bins;
  auto* cursors = offsets + bins + 1;
  result        = {offsets, static_cast<unsigned*>(allocations[1])};
  if (const auto error = CubDebug(cudaMemsetAsync(counts, 0, bins * sizeof(unsigned), stream)))
  {
    return error;
  }
  const unsigned grid = num_segments / 256 + (num_segments % 256 != 0);
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher(grid, 256, 0, stream)
            .doit(partition_kernel<Capacities, false, BeginIterator, EndIterator>,
                  begins,
                  ends,
                  num_segments,
                  counts,
                  result.segment_ids)))
    {
      return error;
    }
  }
  if (const auto error =
        CubDebug(launcher(1, 1, 0, stream).doit(segment_prefix_kernel<Capacities>, counts, offsets, cursors)))
  {
    return error;
  }
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher(grid, 256, 0, stream)
            .doit(partition_kernel<Capacities, true, BeginIterator, EndIterator>,
                  begins,
                  ends,
                  num_segments,
                  cursors,
                  result.segment_ids)))
    {
      return error;
    }
  }
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }
  return CubDebug(DebugSyncStream(stream));
}

// Configure shared scratch, validate occupancy, and launch one entry with a policy-sized grid.
template <typename Entry, typename Key, typename BeginIterator, typename EndIterator, typename Launcher>
CUB_RUNTIME_FUNCTION cudaError_t launch_entry(
  partition_result bins,
  BeginIterator begins,
  EndIterator ends,
  const Key* in,
  Key* out,
  int sm_count,
  cudaStream_t stream,
  Launcher launcher)
{
  constexpr auto kernel  = sort_entry_kernel<Entry, Key, BeginIterator, EndIterator>;
  constexpr size_t bytes = Entry::implementation::template scratch_bytes<Key>();
  // Explicit opt-in is required by the larger merge entries. Infeasible tunings fail.
  if (const auto error =
        CubDebug(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int{bytes})))
  {
    return error;
  }
  int occupancy = 0;
  if (const auto error = CubDebug(launcher.MaxSmOccupancy(occupancy, kernel, Entry::tuning.block_threads, bytes)))
  {
    return error;
  }
  // Zero resident blocks means this kernel cannot execute with the requested resources.
  if (!occupancy)
  {
    return cudaErrorInvalidConfiguration;
  }
  const unsigned long long blocks = static_cast<unsigned long long>(sm_count) * Entry::tuning.blocks_per_sm;
  if (const auto error = CubDebug(launcher(static_cast<unsigned>(blocks), Entry::tuning.block_threads, bytes, stream)
                                    .doit(kernel, bins, begins, ends, in, out)))
  {
    return error;
  }
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }
  return CubDebug(DebugSyncStream(stream));
}

// Launch each entry in order, shifting its offset view while retaining the shared ID queue.
// The index sequence enumerates entry types at compile time.
template <typename Plan,
          typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename Launcher,
          ::cuda::std::size_t... EntryIndices>
CUB_RUNTIME_FUNCTION cudaError_t launch_entries(
  partition_result bins,
  BeginIterator begins,
  EndIterator ends,
  const Key* in,
  Key* out,
  cudaStream_t stream,
  Launcher launcher,
  ::cuda::std::index_sequence<EntryIndices...>)
{
  int sm_count = 0;
  if (const auto error = CubDebug(launcher.MultiProcessorCount(sm_count)))
  {
    return error;
  }
  cudaError_t error = cudaSuccess;
  auto launch       = [&](auto index) {
    constexpr int i = decltype(index)::value;
    if (error == cudaSuccess)
    {
      error = launch_entry<typename Plan::template entry<i>>(
        {bins.bin_offsets + Plan::bin_begin(i), bins.segment_ids}, begins, ends, in, out, sm_count, stream, launcher);
    }
  };
  (launch(::cuda::std::integral_constant<int, EntryIndices>{}), ...);
  return error;
}

// Internal bounded ascending keys-only backend. Oversized segments are left
// unsorted in the overflow ID range; overflow sorting belongs to a later change.
// Its entry composition and policies are fixed here, with no public API route.
template <typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename Launcher = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION cudaError_t dispatch(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  unsigned num_segments,
  const Key* in,
  Key* out,
  partition_result& result,
  cudaStream_t stream = 0,
  Launcher launcher   = {})
{
  static_assert(sizeof(Key) <= 8, "Keys wider than 8 bytes require an architecture-specific GroupMerge policy");
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher.PtxComputeCap(cc)))
  {
    return error;
  }
  return dispatch_compute_cap(group_merge_policy_selector<Key>{}, cc, [&](auto getter) -> cudaError_t {
    using plan = typename group_merge_entries<decltype(getter)>::type;
    if (const auto error = dispatch_partition<typename plan::capacities>(
          temp_storage, temp_storage_bytes, begins, ends, num_segments, result, stream, launcher))
    {
      return error;
    }
    if (!temp_storage || !num_segments)
    {
      return cudaSuccess;
    }
    return launch_entries<plan>(
      result, begins, ends, in, out, stream, launcher, ::cuda::std::make_index_sequence<plan::size>{});
  });
}
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
