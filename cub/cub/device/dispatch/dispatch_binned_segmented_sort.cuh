// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assemble private sorting kernels, manage temporary storage, and launch partitioning followed by sorting.

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
#include <cub/device/dispatch/kernels/kernel_binned_segmented_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Each launch stores its implementation and selector, independently of the selected architecture.
template <typename Key>
using group_merge_sort_kernels_t =
  sort_kernel_list_t<sort_kernel_t<group_merge_implementation, group_merge_warp_policy_selector>,
                     sort_kernel_t<group_merge_implementation, group_merge_small_policy_selector>,
                     sort_kernel_t<group_merge_implementation, group_merge_large_policy_selector<Key>>>;

// Query or use temporary storage, then count, prefix, and scatter IDs into capacity bins.
// These pointers alias caller-provided global storage. It must remain valid until
// every operation queued on the stream has completed, as with other CUB algorithms.
// Empty ranges are skipped; singletons use the same sorting path as other ranges.
template <typename CapacitySelector,
          typename ComputeCapabilityGetter,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
_CCCL_HOST_API cudaError_t dispatch_partition(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  unsigned num_segments,
  partition_result& result,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {})
{
  using capacities_t       = typename CapacitySelector::template capacities_t<ComputeCapabilityGetter>;
  constexpr int total_bins = capacities_t::size + 1;
  // One allocation holds counts, offsets (with an end sentinel), and cursors; the other holds IDs.
  const size_t sizes[] = {sizeof(unsigned) * (3 * total_bins + 1), sizeof(unsigned) * size_t{num_segments}};
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
  auto* offsets = counts + total_bins;
  auto* cursors = offsets + total_bins + 1;
  result        = {offsets, static_cast<unsigned*>(allocations[1])};
  if (const auto error = CubDebug(cudaMemsetAsync(counts, 0, total_bins * sizeof(unsigned), stream)))
  {
    return error;
  }
  const unsigned grid_size = num_segments / 256 + (num_segments % 256 != 0);
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, 256, 0, stream)
            .doit(partition_kernel<CapacitySelector, false, BeginIterator, EndIterator>,
                  begins,
                  ends,
                  num_segments,
                  counts,
                  result.segment_ids)))
    {
      return error;
    }
  }
  if (const auto error = CubDebug(
        launcher_factory(1, 1, 0, stream).doit(segment_prefix_kernel<CapacitySelector>, counts, offsets, cursors)))
  {
    return error;
  }
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, 256, 0, stream)
            .doit(partition_kernel<CapacitySelector, true, BeginIterator, EndIterator>,
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

// Configure shared scratch, validate occupancy, and launch one sorting kernel with a policy-sized grid.
template <typename Kernel,
          typename ComputeCapabilityGetter,
          typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory>
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
  constexpr auto kernel               = sort_kernel<Kernel, Key, BeginIterator, EndIterator>;
  using implementation_t              = typename Kernel::template implementation_t<ComputeCapabilityGetter>;
  constexpr auto tuning               = implementation_t::policy;
  constexpr size_t dynamic_smem_bytes = implementation_t::template scratch_bytes<Key>();
  // Explicit opt-in is required by the larger merge kernels. Infeasible tunings fail.
  if (const auto error =
        CubDebug(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int{dynamic_smem_bytes})))
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
  const unsigned long long grid_size = static_cast<unsigned long long>(sm_count) * tuning.blocks_per_sm;
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
          typename ComputeCapabilityGetter,
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
            ? launch_sort_kernel<typename Kernels::template kernel_t<KernelIndices>, ComputeCapabilityGetter>(
                {bins.bin_offsets + Kernels::template kernel_bin_begin_idx<ComputeCapabilityGetter>(KernelIndices),
                 bins.segment_ids},
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
  unsigned num_segments,
  const Key* in,
  Key* out,
  partition_result& result,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {})
{
  static_assert(sizeof(Key) <= 8, "Keys wider than 8 bytes require an architecture-specific GroupMerge policy");
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }
  return dispatch_compute_cap(group_merge_tuning_arch_selector{}, cc, [&](auto arch) -> cudaError_t {
    using compute_capability_getter_t = decltype(arch);
    if constexpr (compute_capability_getter_t{}() < ::cuda::compute_capability{9, 0})
    {
      return cudaErrorNotSupported;
    }
    else
    {
      using kernels_t = group_merge_sort_kernels_t<Key>;
      if (const auto error = dispatch_partition<kernels_t, compute_capability_getter_t>(
            temp_storage, temp_storage_bytes, begins, ends, num_segments, result, stream, launcher_factory))
      {
        return error;
      }
      if (!temp_storage || !num_segments)
      {
        return cudaSuccess;
      }
      return launch_sort_kernel_list<kernels_t, compute_capability_getter_t>(
        result, begins, ends, in, out, stream, launcher_factory, ::cuda::std::make_index_sequence<kernels_t::size>{});
    }
  });
}
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
