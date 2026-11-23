// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assemble private sorting kernels, manage temporary storage, and launch partitioning followed by sorting.

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_sort_group_merge.cuh>
#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/device_double_buffer.cuh>
#include <cub/detail/double_buffer.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/segmented_sort_lrb.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{
// Shared architecture and key-width eligibility for internal dispatch and its tests.
template <typename Key>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool can_use_lrb(::cuda::compute_capability cc) noexcept
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
  const size_t sizes[] = {sizeof(unsigned) * ((3 * total_bins) + 1), sizeof(unsigned) * size_t{num_segments}};
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
  const unsigned threads_per_block = static_cast<unsigned>(tuning.threads_per_block);
  const unsigned grid_size         = ::cuda::ceil_div(num_segments, threads_per_block);
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, threads_per_block, 0, stream)
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
  if (const auto error = CubDebug(launcher_factory(1, detail::warp_threads, 0, stream)
                                    .doit(segment_prefix_kernel<Capacities>, counts, offsets, cursors)))
  {
    return error;
  }
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, threads_per_block, 0, stream)
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
          SortOrder Order,
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
  constexpr auto kernel               = sort_kernel<Kernel, Order, Key, BeginIterator, EndIterator>;
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
        CubDebug(launcher_factory.MaxSmOccupancy(occupancy, kernel, tuning.threads_per_block, dynamic_smem_bytes)))
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
        launcher_factory(static_cast<unsigned>(grid_size), tuning.threads_per_block, dynamic_smem_bytes, stream)
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
          SortOrder Order,
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
        ? launch_sort_kernel<typename Kernels::template kernel_t<KernelIndices>, Order>(
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

// Sort the overflow ID range with the same per-segment radix kernel as DeviceSegmentedSort.
template <SortOrder Order, typename Key, typename BeginIterator, typename EndIterator, typename KernelLauncherFactory>
_CCCL_HOST_API cudaError_t launch_radix_overflow(
  const unsigned* segment_ids,
  const unsigned* offsets,
  BeginIterator begins,
  EndIterator ends,
  const Key* in,
  Key* out,
  device_double_buffer<Key> radix_buffers,
  int threads_per_block,
  cudaStream_t stream,
  KernelLauncherFactory launcher_factory)
{
  unsigned bounds[2];
  if (const auto error =
        CubDebug(launcher_factory.MemcpyAsync(bounds, offsets, sizeof(bounds), cudaMemcpyDeviceToHost, stream)))
  {
    return error;
  }
  if (const auto error = CubDebug(_CUB::SyncStream(stream)))
  {
    return error;
  }
  using selector_t      = segmented_sort::policy_selector_from_types<Key, NullType>;
  using offset_t        = choose_signed_offset_t<common_iterator_value_t<BeginIterator, EndIterator>>;
  constexpr auto kernel = segmented_sort::
    DeviceSegmentedSortKernelLarge<Order, selector_t, Key, NullType, BeginIterator, EndIterator, offset_t>;
  for (unsigned i = bounds[0]; i < bounds[1];)
  {
    const unsigned grid_size = (::cuda::std::min) (bounds[1] - i, unsigned{(::cuda::std::numeric_limits<int>::max)()});
    if (const auto error = CubDebug(
          launcher_factory(grid_size, threads_per_block, 0, stream)
            .doit(kernel,
                  segment_ids + i,
                  in,
                  out,
                  radix_buffers,
                  static_cast<const NullType*>(nullptr),
                  static_cast<NullType*>(nullptr),
                  device_double_buffer<NullType>(nullptr, nullptr),
                  begins,
                  ends)))
    {
      return error;
    }
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }
    i += grid_size;
  }
  return CubDebug(DebugSyncStream(stream));
}

// Host-launched keys-only backend. GroupMerge handles bounded bins and radix handles overflow.
// num_items is the size of both key buffers and must cover every exclusive end offset because radix passes use the same
// offsets in scratch. When overflow is possible, dispatch reads the overflow range back to the host and synchronizes
// the stream, which prevents graph capture.
template <SortOrder Order,
          typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
          typename PolicySelector        = policy_selector<Key>>
_CCCL_HOST_API cudaError_t dispatch(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  segmented_sort::global_segment_offset_t num_segments,
  ::cuda::std::int64_t num_items,
  DoubleBuffer<Key>& keys,
  bool is_overwrite_okay,
  partition_result& result,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {},
  int max_segments_per_invocation        = (::cuda::std::numeric_limits<int>::max)())
{
  if (num_segments < 0 || num_items < 0 || max_segments_per_invocation <= 0)
  {
    return cudaErrorInvalidValue;
  }
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
      using policy_getter_t  = decltype(policy_getter);
      using kernels_t        = group_merge_sort_kernels_t<Key, PolicySelector>;
      using capacities_t     = typename kernels_t::template capacities_t<policy_getter_t>;
      const auto window_size = [&](segmented_sort::global_segment_offset_t remaining) {
        return static_cast<unsigned>(
          (::cuda::std::min) (remaining, segmented_sort::global_segment_offset_t{max_segments_per_invocation}));
      };
      const auto partition =
        [&](void* storage, size_t& bytes, BeginIterator current_begins, EndIterator current_ends, unsigned count) {
          return dispatch_partition<capacities_t, partition_subpolicy_selector<PolicySelector>>(
            storage,
            bytes,
            current_begins,
            current_ends,
            count,
            result,
            stream,
            launcher_factory,
            policy_getter().partition);
        };
      size_t partition_bytes = 0;
      if (const auto error = partition(nullptr, partition_bytes, begins, ends, window_size(num_segments)))
      {
        return error;
      }
      const bool may_have_overflow = num_segments && num_items > capacities_t::bounds.back();
      const size_t sizes[]         = {
        partition_bytes, may_have_overflow && !is_overwrite_okay ? static_cast<size_t>(num_items) * sizeof(Key) : 0};
      void* allocations[2]{};
      if (const auto error = CubDebug(alias_temporaries(temp_storage, temp_storage_bytes, allocations, sizes)))
      {
        return error;
      }
      if (!temp_storage)
      {
        return cudaSuccess;
      }
      if (!num_segments || !num_items)
      {
        // Publish empty bin offsets even when there are no keys to sort.
        return partition(allocations[0], partition_bytes, begins, ends, 0);
      }
      const auto policy        = segmented_sort::policy_selector_from_types<Key, NullType>{}(cc).large_segment;
      const int passes         = ::cuda::ceil_div(int{sizeof(Key) * 8}, policy.radix_bits);
      const int final_selector = (keys.selector + (is_overwrite_okay ? passes : 1)) % 2;
      Key* out                 = keys.d_buffers[final_selector];
      Key* other = is_overwrite_okay ? keys.d_buffers[1 - final_selector] : static_cast<Key*>(allocations[1]);
      const device_double_buffer<Key> radix_buffers(passes % 2 ? out : other, passes % 2 ? other : out);

      // IDs index the current offset iterators. The offsets themselves still index the full key buffers.
      for (segmented_sort::global_segment_offset_t current_seg_offset = 0; current_seg_offset < num_segments;)
      {
        const unsigned current_num_segments = window_size(num_segments - current_seg_offset);
        const auto current_begins           = begins + current_seg_offset;
        const auto current_ends             = ends + current_seg_offset;
        if (const auto error =
              partition(allocations[0], partition_bytes, current_begins, current_ends, current_num_segments))
        {
          return error;
        }
        if (may_have_overflow)
        {
          if (const auto error = launch_radix_overflow<Order>(
                result.segment_ids,
                result.bin_offsets + capacities_t::size,
                current_begins,
                current_ends,
                keys.Current(),
                out,
                radix_buffers,
                policy.threads_per_block,
                stream,
                launcher_factory))
          {
            return error;
          }
        }
        if (const auto error = launch_sort_kernel_list<kernels_t, Order>(
              result,
              current_begins,
              current_ends,
              keys.Current(),
              out,
              stream,
              launcher_factory,
              ::cuda::std::make_index_sequence<kernels_t::size>{},
              policy_getter))
        {
          return error;
        }
        current_seg_offset += current_num_segments;
      }
      keys.selector = final_selector;
      return cudaSuccess;
    });
  }
}

template <SortOrder Order,
          typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
          typename PolicySelector        = policy_selector<Key>>
_CCCL_HOST_API cudaError_t dispatch(
  void* temp_storage,
  size_t& temp_storage_bytes,
  BeginIterator begins,
  EndIterator ends,
  segmented_sort::global_segment_offset_t num_segments,
  ::cuda::std::int64_t num_items,
  const Key* in,
  Key* out,
  partition_result& result,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {})
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  DoubleBuffer<Key> keys(const_cast<Key*>(in), out);
  return dispatch<Order, Key, BeginIterator, EndIterator, KernelLauncherFactory, PolicySelector>(
    temp_storage,
    temp_storage_bytes,
    begins,
    ends,
    num_segments,
    num_items,
    keys,
    false,
    result,
    stream,
    launcher_factory);
}
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END
