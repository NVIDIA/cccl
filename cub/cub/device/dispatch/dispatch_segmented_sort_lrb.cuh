// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assemble private sorting kernels, manage temporary storage, and launch the tiny-segment walk and partitioning
// followed by sorting.

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
#include <cub/detail/segmented_sort_entries.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Shared architecture and key-width eligibility for internal dispatch and its tests.
template <typename Key>
_CCCL_HOST_DEVICE_API constexpr bool can_use_lrb(::cuda::compute_capability cc)
{
  return sizeof(Key) <= 8 && cc >= ::cuda::compute_capability{9, 0};
}

// Each launch stores its implementation and selector, independently of the selected architecture.
template <typename Key>
using group_merge_sort_kernels_t =
  sort_kernel_list_t<sort_kernel_t<group_merge_implementation, group_merge_warp_policy_selector>,
                     sort_kernel_t<group_merge_implementation, group_merge_small_policy_selector>,
                     sort_kernel_t<group_merge_implementation, group_merge_large_policy_selector<Key>>>;

// Query or use temporary storage, then sort tiny segments while counting and queueing the rest, prefix, and
// scatter the queued IDs into capacity bins. These pointers alias caller-provided global storage. It must remain valid
// until every operation queued on the stream has completed, as with other CUB algorithms.
// Empty ranges are skipped; segments shorter than tiny_walk_policy::inline_cutoff are written to out here.
// Segments must not overlap and must lie within the num_items keys of in, as CUB requires.
template <typename Capacities,
          typename BeginIterator,
          typename EndIterator,
          typename Key,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
_CCCL_HOST_API cudaError_t dispatch_tiny_sort_and_partition(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  ::cuda::std::int64_t num_items,
  unsigned num_segments,
  const Key* in,
  Key* out,
  partition_result& result,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {})
{
  constexpr int total_bins = Capacities::size + 1;
  // Non-overlapping segments within num_items keys hold at most num_items / inline_cutoff deferred segments.
  const auto max_deferred = static_cast<unsigned>(
    (::cuda::std::min) (::cuda::std::int64_t{num_segments},
                        (::cuda::std::max) (num_items, ::cuda::std::int64_t{0}) / tiny_walk_policy::inline_cutoff));
  // The first allocation holds counts and the deferred count (cleared together), offsets (with an end
  // sentinel), and cursors. The others hold binned IDs and the walk's deferred IDs.
  const size_t sizes[] = {sizeof(unsigned) * (3 * total_bins + 2),
                          sizeof(unsigned) * size_t{max_deferred},
                          sizeof(unsigned) * size_t{max_deferred}};
  void* allocations[3]{};
  if (const auto error = CubDebug(alias_temporaries(temp_storage, temp_storage_bytes, allocations, sizes)))
  {
    return error;
  }
  if (!temp_storage)
  {
    return cudaSuccess;
  }
  auto* counts         = static_cast<unsigned*>(allocations[0]);
  auto* deferred_count = counts + total_bins;
  auto* offsets        = deferred_count + 1;
  auto* cursors        = offsets + total_bins + 1;
  auto* deferred_ids   = static_cast<unsigned*>(allocations[2]);
  result               = {offsets, static_cast<unsigned*>(allocations[1])};
  if (const auto error = CubDebug(cudaMemsetAsync(counts, 0, (total_bins + 1) * sizeof(unsigned), stream)))
  {
    return error;
  }
  constexpr unsigned walk_threads  = tiny_walk_policy::block_threads;
  constexpr unsigned walk_segments = walk_threads * tiny_walk_policy::segments_per_thread;
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(num_segments / walk_segments + (num_segments % walk_segments != 0), walk_threads, 0, stream)
            .doit(tiny_walk_kernel<Capacities, BeginIterator, EndIterator, Key>,
                  begins,
                  ends,
                  num_segments,
                  in,
                  out,
                  counts,
                  deferred_count,
                  deferred_ids)))
    {
      return error;
    }
  }
  if (const auto error =
        CubDebug(launcher_factory(1, 1, 0, stream).doit(segment_prefix_kernel<Capacities>, counts, offsets, cursors)))
  {
    return error;
  }
  if (max_deferred)
  {
    int sm_count = 0;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
    {
      return error;
    }
    // The deferred count is not known on the host; the grid strides over it and never exceeds one block per
    // block_threads possible deferred segments.
    constexpr unsigned block_threads = partition_policy::block_threads;
    const unsigned grid_size = (::cuda::std::min) (static_cast<unsigned>(sm_count) * partition_policy::blocks_per_sm,
                                                   max_deferred / block_threads + (max_deferred % block_threads != 0));
    if (const auto error = CubDebug(
          launcher_factory(grid_size, block_threads, 0, stream)
            .doit(compact_scatter_kernel<Capacities, BeginIterator, EndIterator>,
                  begins,
                  ends,
                  deferred_count,
                  deferred_ids,
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

// Configure shared scratch, validate occupancy, and launch one sorting kernel with a policy-sized grid.
template <typename Kernel, int CC, typename Key, typename BeginIterator, typename EndIterator, typename KernelLauncherFactory>
_CCCL_HOST_API cudaError_t launch_sort_kernel(
  partition_result bins,
  BeginIterator begins,
  EndIterator ends,
  const Key* in,
  Key* out,
  int sm_count,
  cudaStream_t stream,
  KernelLauncherFactory launcher_factory)
{
  constexpr auto kernel = sort_kernel<Kernel, Key, BeginIterator, EndIterator>;
  using implementation_t =
    typename Kernel::template implementation_t<policy_getter<typename Kernel::policy_selector, CC>>;
  constexpr auto tuning               = implementation_t::policy;
  constexpr size_t dynamic_smem_bytes = implementation_t::template scratch_bytes<Key>();
  // Explicit opt-in is required by the larger merge kernels. Infeasible tunings fail.
  if (const auto error = launcher_factory.set_max_dynamic_smem_size_for(kernel, int{dynamic_smem_bytes}))
  {
    return error;
  }
  int occupancy = 0;
  if (const auto error =
        CubDebug(launcher_factory.MaxSmOccupancy(occupancy, kernel, tuning.block_threads, dynamic_smem_bytes)))
  {
    return error;
  }
  // Zero resident blocks means this kernel cannot execute with the requested resources.
  if (!occupancy)
  {
    return cudaErrorInvalidConfiguration;
  }
  const unsigned long long grid_size = static_cast<unsigned long long>(sm_count) * tuning.grid_size_multiplier;
  if (const auto error = CubDebug(
        launcher_factory(static_cast<unsigned>(grid_size), tuning.block_threads, dynamic_smem_bytes, stream)
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

// Launch each kernel in order, shifting its offset view while retaining the shared ID queue.
// The index sequence enumerates kernel types at compile time.
template <typename Kernels,
          int CC,
          typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory,
          ::cuda::std::size_t... KernelIndices>
_CCCL_HOST_API cudaError_t launch_sort_kernel_list(
  partition_result bins,
  BeginIterator begins,
  EndIterator ends,
  const Key* in,
  Key* out,
  cudaStream_t stream,
  KernelLauncherFactory launcher_factory,
  ::cuda::std::index_sequence<KernelIndices...>)
{
  int sm_count = 0;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }
  cudaError_t error = cudaSuccess;
  ((error = error == cudaSuccess
            ? launch_sort_kernel<typename Kernels::template kernel_t<KernelIndices>, CC>(
                {bins.bin_offsets + Kernels::template kernel_bin_begin_idx<CC>(KernelIndices), bins.segment_ids},
                begins,
                ends,
                in,
                out,
                sm_count,
                stream,
                launcher_factory)
            : error),
   ...);
  return error;
}

// Host-launched internal bounded ascending keys-only backend; device-side API calls are not supported.
// Oversized segments are left unsorted in the overflow ID range; overflow sorting belongs to a later change.
// num_items bounds the ID buffers, so segments must not overlap and must lie within the num_items keys of in.
// Its kernel list and policies are fixed here, with no public API route.
template <typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
_CCCL_HOST_API cudaError_t dispatch(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  ::cuda::std::int64_t num_items,
  unsigned num_segments,
  const Key* in,
  Key* out,
  partition_result& result,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }
  return dispatch_compute_cap(group_merge_tuning_arch_selector{}, cc, [&](auto arch) -> cudaError_t {
    constexpr auto selected_cc = decltype(arch){}();
    if constexpr (!can_use_lrb<Key>(selected_cc))
    {
      return cudaErrorNotSupported;
    }
    else
    {
      using kernels_t    = group_merge_sort_kernels_t<Key>;
      using capacities_t = typename kernels_t::template capacities_t<policy_getter, selected_cc.get()>;
      static_assert(capacities_t::bounds[0] >= tiny_walk_policy::inline_cutoff,
                    "The first sorting bin must start at the tiny-walk cutoff; shorter bins would stay empty");
      if (const auto error = dispatch_tiny_sort_and_partition<capacities_t>(
            temp_storage,
            temp_storage_bytes,
            begins,
            ends,
            num_items,
            num_segments,
            in,
            out,
            result,
            stream,
            launcher_factory))
      {
        return error;
      }
      if (!temp_storage || !num_segments)
      {
        return cudaSuccess;
      }
      return launch_sort_kernel_list<kernels_t, selected_cc.get()>(
        result, begins, ends, in, out, stream, launcher_factory, ::cuda::std::make_index_sequence<kernels_t::size>{});
    }
  });
}
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
