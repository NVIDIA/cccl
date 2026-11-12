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
#include <cub/detail/segmented_sort_lrb.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/__utility/integer_sequence.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{
// Shared architecture and key-width eligibility for internal dispatch and its tests.
template <typename Key>
_CCCL_HOST_DEVICE_API constexpr bool can_use_lrb(::cuda::compute_capability cc)
{
  return sizeof(Key) <= 8 && cc >= ::cuda::compute_capability{9, 0};
}

// Each launch stores its implementation and selector, independently of the selected architecture.
template <typename Key, typename PolicySelector = policy_selector<Key>>
using group_merge_sort_kernels_t = sort_kernel_list_t<
  sort_kernel_t<agent_group_merge, group_merge_policy_selector<PolicySelector, &group_merge_policies::warp>>,
  sort_kernel_t<agent_group_merge, group_merge_policy_selector<PolicySelector, &group_merge_policies::small_block>>,
  sort_kernel_t<agent_group_merge, group_merge_policy_selector<PolicySelector, &group_merge_policies::large>>>;

// Query or use temporary storage, then count, prefix, and scatter IDs into capacity bins.
// These pointers alias caller-provided global storage. It must remain valid until
// every operation queued on the stream has completed, as with other CUB algorithms.
// Empty ranges are skipped; singletons use the same sorting path as other ranges.
template <typename Capacities,
          typename PartitionPolicySelector = partition_policy_selector,
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
  KernelLauncherFactory launcher_factory = {},
  partition_policy tuning                = {})
{
  constexpr int total_bins = Capacities::size + 1;
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
  const unsigned block_threads = static_cast<unsigned>(tuning.block_threads);
  const unsigned grid_size     = num_segments / block_threads + (num_segments % block_threads != 0);
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, block_threads, 0, stream)
            .doit(partition_kernel<Capacities, false, BeginIterator, EndIterator, PartitionPolicySelector>,
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
        CubDebug(launcher_factory(1, 1, 0, stream).doit(segment_prefix_kernel<Capacities>, counts, offsets, cursors)))
  {
    return error;
  }
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, block_threads, 0, stream)
            .doit(partition_kernel<Capacities, true, BeginIterator, EndIterator, PartitionPolicySelector>,
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
          typename PolicyGetter,
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
  KernelLauncherFactory launcher_factory,
  PolicyGetter)
{
  constexpr auto kernel               = sort_kernel<Kernel, Key, BeginIterator, EndIterator>;
  using implementation_t              = typename Kernel::template implementation_t<PolicyGetter>;
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
          typename PolicyGetter,
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
  ::cuda::std::index_sequence<KernelIndices...>,
  PolicyGetter)
{
  int sm_count = 0;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }
  cudaError_t error = cudaSuccess;
  ((error =
      error == cudaSuccess
        ? launch_sort_kernel<typename Kernels::template kernel_t<KernelIndices>>(
            {bins.bin_offsets + Kernels::template kernel_bin_begin_idx<PolicyGetter>(KernelIndices), bins.segment_ids},
            begins,
            ends,
            in,
            out,
            sm_count,
            stream,
            launcher_factory,
            typename Kernels::template kernel_t<KernelIndices>::template selected_policy_getter<PolicyGetter>{})
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
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
          typename PolicySelector        = policy_selector<Key>>
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
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }
  if constexpr (sizeof(Key) > 8)
  {
    return cudaErrorNotSupported;
  }
  else
  {
    if (!can_use_lrb<Key>(cc))
    {
      return cudaErrorNotSupported;
    }
    return dispatch_compute_cap(PolicySelector{}, cc, [&](auto policy_getter) -> cudaError_t {
      using policy_getter_t = decltype(policy_getter);
      using kernels_t       = group_merge_sort_kernels_t<Key, PolicySelector>;
      using capacities_t    = typename kernels_t::template capacities_t<policy_getter_t>;
      if (const auto error = dispatch_partition<capacities_t, partition_subpolicy_selector<PolicySelector>>(
            temp_storage,
            temp_storage_bytes,
            begins,
            ends,
            num_segments,
            result,
            stream,
            launcher_factory,
            policy_getter().partition))
      {
        return error;
      }
      if (!temp_storage || !num_segments)
      {
        return cudaSuccess;
      }
      return launch_sort_kernel_list<kernels_t>(
        result,
        begins,
        ends,
        in,
        out,
        stream,
        launcher_factory,
        ::cuda::std::make_index_sequence<kernels_t::size>{},
        policy_getter);
    });
  }
}
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END
