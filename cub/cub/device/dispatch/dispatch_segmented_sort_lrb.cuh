// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_sort_lrb.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{

template <typename Policy, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename CountT>
struct DeviceSegmentedSortLrbKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    CountKernel,
    DeviceSegmentedSortLrbCountKernel<Policy, BeginOffsetIteratorT, EndOffsetIteratorT, CountT>);

  CUB_DEFINE_KERNEL_GETTER(PrefixKernel, DeviceSegmentedSortLrbPrefixKernel<Policy, CountT>);

  CUB_DEFINE_KERNEL_GETTER(
    ScatterKernel,
    DeviceSegmentedSortLrbScatterKernel<Policy, BeginOffsetIteratorT, EndOffsetIteratorT, CountT>);
};

template <typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename CountT         = ::cuda::std::uint32_t,
          typename Policy         = lrb_policy<>,
          typename KernelSource   = DeviceSegmentedSortLrbKernelSource<Policy, BeginOffsetIteratorT, EndOffsetIteratorT, CountT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchSegmentedSortLrb
{
  void* d_temp_storage;
  size_t& temp_storage_bytes;
  BeginOffsetIteratorT d_begin_offsets;
  EndOffsetIteratorT d_end_offsets;
  int num_segments;
  lrb_plan<CountT>& plan;
  cudaStream_t stream;
  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static size_t metadata_bytes()
  {
    // counts + exclusive offsets + atomic offset copies + group offsets + summary
    const size_t counts =
      sizeof(CountT)
      * (Policy::warp_bin_count + Policy::small_block_bin_count + Policy::large_block_bin_count + Policy::overflow_bin_count);
    const size_t offsets =
      sizeof(CountT)
      * ((Policy::warp_bin_count + 1) + (Policy::small_block_bin_count + 1) + (Policy::large_block_bin_count + 1)
         + (Policy::overflow_bin_count + 1));
    const size_t atomics  = offsets;
    const size_t groups   = sizeof(CountT)
                      * ((Policy::warp_bin_count + 1) + (Policy::small_block_bin_count + 1)
                         + (Policy::large_block_bin_count + 1));
    const size_t summary = sizeof(lrb_summary<CountT>);
    return counts + offsets + atomics + groups + summary;
  }

  template <typename CountKernelT, typename PrefixKernelT, typename ScatterKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(CountKernelT count_kernel, PrefixKernelT prefix_kernel, ScatterKernelT scatter_kernel)
  {
    constexpr int num_allocs = 2;
    size_t allocation_sizes[num_allocs]{};
    allocation_sizes[0] = metadata_bytes();
    allocation_sizes[1] = static_cast<size_t>(num_segments) * sizeof(CountT);

    void* allocations[num_allocs]{};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      return cudaSuccess;
    }

    CountT* cursor = reinterpret_cast<CountT*>(allocations[0]);

    CountT* d_warp_counts        = cursor;
    cursor += Policy::warp_bin_count;
    CountT* d_small_block_counts = cursor;
    cursor += Policy::small_block_bin_count;
    CountT* d_large_block_counts = cursor;
    cursor += Policy::large_block_bin_count;
    CountT* d_overflow_counts    = cursor;
    cursor += Policy::overflow_bin_count;

    CountT* d_warp_offsets = cursor;
    cursor += Policy::warp_bin_count + 1;
    CountT* d_small_block_offsets = cursor;
    cursor += Policy::small_block_bin_count + 1;
    CountT* d_large_block_offsets = cursor;
    cursor += Policy::large_block_bin_count + 1;
    CountT* d_overflow_offsets = cursor;
    cursor += Policy::overflow_bin_count + 1;

    CountT* d_warp_offsets_atomic = cursor;
    cursor += Policy::warp_bin_count + 1;
    CountT* d_small_block_offsets_atomic = cursor;
    cursor += Policy::small_block_bin_count + 1;
    CountT* d_large_block_offsets_atomic = cursor;
    cursor += Policy::large_block_bin_count + 1;
    CountT* d_overflow_offsets_atomic = cursor;
    cursor += Policy::overflow_bin_count + 1;

    CountT* d_warp_group_offsets = cursor;
    cursor += Policy::warp_bin_count + 1;
    CountT* d_small_block_group_offsets = cursor;
    cursor += Policy::small_block_bin_count + 1;
    CountT* d_large_block_group_offsets = cursor;
    cursor += Policy::large_block_bin_count + 1;

    auto* d_summary = reinterpret_cast<lrb_summary<CountT>*>(cursor);

    plan.d_segment_ids                = static_cast<CountT*>(allocations[1]);
    plan.d_warp_bin_offsets           = d_warp_offsets;
    plan.d_small_block_bin_offsets    = d_small_block_offsets;
    plan.d_large_block_bin_offsets    = d_large_block_offsets;
    plan.d_overflow_offsets           = d_overflow_offsets;
    plan.d_warp_group_offsets         = d_warp_group_offsets;
    plan.d_small_block_group_offsets  = d_small_block_group_offsets;
    plan.d_large_block_group_offsets  = d_large_block_group_offsets;
    plan.d_summary                    = d_summary;

    if (num_segments == 0)
    {
      if (const auto error = CubDebug(::cudaMemsetAsync(allocations[0], 0, allocation_sizes[0], stream)))
      {
        return error;
      }
      return CubDebug(detail::DebugSyncStream(stream));
    }

    if (const auto error = CubDebug(::cudaMemsetAsync(allocations[0], 0, allocation_sizes[0], stream)))
    {
      return error;
    }

    const int grid = static_cast<int>(::cuda::ceil_div(num_segments, Policy::block_threads));

    if (const auto error = CubDebug(
          launcher_factory(grid, Policy::block_threads, 0, stream)
            .doit(count_kernel,
                  d_begin_offsets,
                  d_end_offsets,
                  num_segments,
                  d_warp_counts,
                  d_small_block_counts,
                  d_large_block_counts,
                  d_overflow_counts)))
    {
      return error;
    }
    if (const auto error = CubDebug(::cudaPeekAtLastError()))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    if (const auto error = CubDebug(
          launcher_factory(1, 1, 0, stream)
            .doit(prefix_kernel,
                  d_warp_counts,
                  d_small_block_counts,
                  d_large_block_counts,
                  d_overflow_counts,
                  d_warp_offsets,
                  d_small_block_offsets,
                  d_large_block_offsets,
                  d_overflow_offsets,
                  d_warp_offsets_atomic,
                  d_small_block_offsets_atomic,
                  d_large_block_offsets_atomic,
                  d_overflow_offsets_atomic,
                  d_warp_group_offsets,
                  d_small_block_group_offsets,
                  d_large_block_group_offsets,
                  d_summary)))
    {
      return error;
    }
    if (const auto error = CubDebug(::cudaPeekAtLastError()))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    if (const auto error = CubDebug(
          launcher_factory(grid, Policy::block_threads, 0, stream)
            .doit(scatter_kernel,
                  d_begin_offsets,
                  d_end_offsets,
                  num_segments,
                  d_warp_offsets_atomic,
                  d_small_block_offsets_atomic,
                  d_large_block_offsets_atomic,
                  d_overflow_offsets_atomic,
                  plan.d_segment_ids)))
    {
      return error;
    }
    if (const auto error = CubDebug(::cudaPeekAtLastError()))
    {
      return error;
    }
    return CubDebug(detail::DebugSyncStream(stream));
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return Invoke(kernel_source.CountKernel(), kernel_source.PrefixKernel(), kernel_source.ScatterKernel());
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    lrb_plan<CountT>& plan,
    cudaStream_t stream                  = 0,
    KernelSource kernel_source           = {},
    KernelLauncherFactory launcher_factory = {})
  {
    DispatchSegmentedSortLrb dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_begin_offsets,
      d_end_offsets,
      num_segments,
      plan,
      stream,
      kernel_source,
      launcher_factory};
    return CubDebug(dispatch.Invoke());
  }
};

} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END
