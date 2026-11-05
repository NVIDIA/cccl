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
#include <cub/detail/device_double_buffer.cuh>
#include <cub/detail/double_buffer.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/segmented_sort_entries.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
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

// Query or use temporary storage, then count, prefix, and scatter IDs into capacity bins.
// These pointers alias caller-provided global storage. It must remain valid until
// every operation queued on the stream has completed, as with other CUB algorithms.
// Empty ranges are skipped; singletons use the same sorting path as other ranges.
template <typename Capacities,
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
  constexpr unsigned block_threads = partition_policy::block_threads;
  const unsigned grid_size         = num_segments / block_threads + (num_segments % block_threads != 0);
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, block_threads, 0, stream)
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
        CubDebug(launcher_factory(1, 1, 0, stream).doit(segment_prefix_kernel<Capacities>, counts, offsets, cursors)))
  {
    return error;
  }
  if (num_segments)
  {
    if (const auto error = CubDebug(
          launcher_factory(grid_size, block_threads, 0, stream)
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

// Configure shared scratch, validate occupancy, and launch one sorting kernel with a policy-sized grid.
template <typename Kernel,
          int CC,
          bool Descending = false,
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
  constexpr auto kernel = sort_kernel<Kernel, Key, BeginIterator, EndIterator, Descending>;
  using implementation_t =
    typename Kernel::template implementation_t<policy_getter<typename Kernel::policy_selector, CC>>;
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
          int CC,
          bool Descending = false,
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
            ? launch_sort_kernel<typename Kernels::template kernel_t<KernelIndices>, CC, Descending>(
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
      if (const auto error = dispatch_partition<capacities_t>(
            temp_storage, temp_storage_bytes, begins, ends, num_segments, result, stream, launcher_factory))
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
// Public routing retains current CUB for pairs, custom tuning, and wider segment counts.
// Device-side callers stay on the existing dispatch because this backend requires host resource setup.
template <typename Key, typename Value, typename PolicySelector>
_CCCL_HOST_DEVICE_API constexpr bool
can_use_lrb(::cuda::compute_capability cc, ::cuda::std::int64_t num_items, ::cuda::std::int64_t num_segments)
{
  return !_CCCL_HAS_CDP() && can_use_lrb<Key>(cc) && ::cuda::std::is_same_v<Value, NullType>
      && ::cuda::std::is_same_v<PolicySelector, segmented_sort::policy_selector_from_types<Key, Value>>
      && num_items >= 0 && num_segments >= 0
      && static_cast<unsigned long long>(num_segments) < ::cuda::std::numeric_limits<unsigned>::max();
}

// Complete keys-only path: bounded GroupMerge bins and CUB's existing radix kernel for overflow.
// Pointer overloads preserve the input with an intermediate array; DoubleBuffer reuses both caller buffers.
// Reading the overflow range synchronizes the stream. Capture and device invocation remain on current CUB.
template <SortOrder Order,
          typename Offset,
          typename Key,
          typename BeginIterator,
          typename EndIterator,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
_CCCL_HOST_API cudaError_t dispatch(
  void* temp_storage,
  size_t& temp_storage_bytes,
  DoubleBuffer<Key>& keys,
  ::cuda::std::int64_t num_items,
  unsigned num_segments,
  BeginIterator begins,
  EndIterator ends,
  bool is_overwrite_okay,
  cudaStream_t stream                    = 0,
  KernelLauncherFactory launcher_factory = {})
{
  using kernels_t        = group_merge_sort_kernels_t<Key>;
  using radix_selector_t = segmented_sort::policy_selector_from_types<Key, NullType>;
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
      using capacities_t = typename kernels_t::template capacities_t<policy_getter, selected_cc.get()>;
      partition_result bins;
      size_t partition_bytes = 0;
      if (const auto error = dispatch_partition<capacities_t>(
            nullptr, partition_bytes, begins, ends, num_segments, bins, stream, launcher_factory))
      {
        return error;
      }
      const size_t sizes[] = {partition_bytes, is_overwrite_okay ? 0 : sizeof(Key) * size_t(num_items)};
      void* allocations[2]{};
      if (const auto error = CubDebug(alias_temporaries(temp_storage, temp_storage_bytes, allocations, sizes)))
      {
        return error;
      }
      if (!temp_storage || !num_segments)
      {
        return cudaSuccess;
      }
      if (const auto error = dispatch_partition<capacities_t>(
            allocations[0], partition_bytes, begins, ends, num_segments, bins, stream, launcher_factory))
      {
        return error;
      }
      // The radix policy is selected independently from the GroupMerge tuning.
      return dispatch_compute_cap(radix_selector_t{}, cc, [&](auto radix_getter) -> cudaError_t {
        const auto radix_policy  = radix_getter().large_segment;
        const int num_passes     = ::cuda::ceil_div(int(sizeof(Key) * 8), radix_policy.radix_bits);
        const bool odd_passes    = num_passes % 2 != 0;
        const int final_selector = is_overwrite_okay ? (keys.selector + num_passes) % 2 : 1 - keys.selector;
        Key* out                 = keys.d_buffers[final_selector];
        Key* intermediate        = static_cast<Key*>(allocations[1]);
        device_double_buffer<Key> radix_buffers(
          (is_overwrite_okay || odd_passes) ? keys.Alternate() : intermediate,
          is_overwrite_okay ? keys.Current() : (odd_passes ? intermediate : keys.Alternate()));
        // Copy only the overflow boundaries; segment IDs stay on the device.
        unsigned overflow_offsets[2]{};
        if (const auto error = CubDebug(cudaMemcpyAsync(
              overflow_offsets,
              bins.bin_offsets + capacities_t::size,
              sizeof(overflow_offsets),
              cudaMemcpyDeviceToHost,
              stream)))
        {
          return error;
        }
        if (const auto error = CubDebug(cudaStreamSynchronize(stream)))
        {
          return error;
        }
        if (const auto error = launch_sort_kernel_list<kernels_t, selected_cc.get(), Order == SortOrder::Descending>(
              bins,
              begins,
              ends,
              keys.Current(),
              out,
              stream,
              launcher_factory,
              ::cuda::std::make_index_sequence<kernels_t::size>{}))
        {
          return error;
        }
        // A CUDA x-grid is int-sized even though segment IDs are unsigned.
        for (unsigned first_segment = overflow_offsets[0]; first_segment < overflow_offsets[1];)
        {
          const unsigned grid_size =
            (::cuda::std::min) (overflow_offsets[1] - first_segment, unsigned(::cuda::std::numeric_limits<int>::max()));
          constexpr auto kernel = segmented_sort::
            DeviceSegmentedSortKernelLarge<Order, radix_selector_t, Key, NullType, BeginIterator, EndIterator, Offset>;
          if (const auto error = CubDebug(
                launcher_factory(grid_size, radix_policy.threads_per_block, 0, stream)
                  .doit(kernel,
                        bins.segment_ids + first_segment,
                        keys.Current(),
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
          first_segment += grid_size;
        }
        keys.selector = final_selector;
        return CubDebug(DebugSyncStream(stream));
      });
    }
  });
}
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
