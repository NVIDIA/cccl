// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/util_namespace.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
enum class worker
{
  block,
  warp,
  thread
};

template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          ForceInclusive EnforceInclusive>
struct device_segmented_scan_kernel_source
{
  CUB_DEFINE_KERNEL_GETTER(
    segmented_scan_kernel,
    device_segmented_scan_kernel<
      MaxPolicyT,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      OffsetT,
      ScanOpT,
      InitValueT,
      AccumT,
      EnforceInclusive == ForceInclusive::Yes>);

  CUB_DEFINE_KERNEL_GETTER(
    warp_segmented_scan_kernel,
    device_warp_segmented_scan_kernel<
      MaxPolicyT,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      OffsetT,
      ScanOpT,
      InitValueT,
      AccumT,
      EnforceInclusive == ForceInclusive::Yes>);

  CUB_DEFINE_KERNEL_GETTER(
    thread_segmented_scan_kernel,
    device_thread_segmented_scan_kernel<
      MaxPolicyT,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      OffsetT,
      ScanOpT,
      InitValueT,
      AccumT,
      EnforceInclusive == ForceInclusive::Yes>);

  CUB_RUNTIME_FUNCTION static constexpr size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};

template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorInputT,
  typename EndOffsetIteratorInputT,
  typename BeginOffsetIteratorOutputT,
  typename ScanOpT,
  typename InitValueT,
  typename AccumT                 = ::cuda::std::__accumulator_t<ScanOpT,
                                                                 cub::detail::it_value_t<InputIteratorT>,
                                                                 ::cuda::std::_If<::cuda::std::is_same_v<InitValueT, NullType>,
                                                                                  cub::detail::it_value_t<InputIteratorT>,
                                                                                  typename InitValueT::value_type>>,
  ForceInclusive EnforceInclusive = ForceInclusive::No,
  typename OffsetT                = typename detail::
    common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>,
  typename PolicyHub = detail::segmented_scan::
    policy_hub<detail::it_value_t<InputIteratorT>, detail::it_value_t<OutputIteratorT>, AccumT, OffsetT, ScanOpT>,
  typename KernelSource = detail::segmented_scan::device_segmented_scan_kernel_source<
    typename PolicyHub::MaxPolicy,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    OffsetT,
    ScanOpT,
    InitValueT,
    AccumT,
    EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct dispatch_segmented_scan
{
  static_assert(::cuda::std::is_integral_v<OffsetT> && sizeof(OffsetT) >= 4 && sizeof(OffsetT) <= 8,
                "dispatch_segmented_scan only supports integral offset types of 4- or 8-bytes");

  /// Device-accessible allocation of temporary storage.  When nullptr, the
  /// required allocation size is written to \p temp_storage_bytes and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of \p d_temp_storage allocation
  size_t& temp_storage_bytes;

  /// Iterator to the input sequence of data items
  InputIteratorT d_in;

  /// Iterator to the output sequence of data items
  OutputIteratorT d_out;

  /// The number of segments that comprise the segmented scan data
  ::cuda::std::int64_t num_segments;

  /// Offsets to beginning of each segment in the input sequence
  BeginOffsetIteratorInputT d_input_begin_offsets;

  /// Offsets to end of each segment in the input sequence
  EndOffsetIteratorInputT d_input_end_offsets;

  /// Offsets to beginning of each segment in the output sequence
  BeginOffsetIteratorOutputT d_output_begin_offsets;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// Number of segments processed by a worker
  int num_segments_per_worker;

  worker worker_choice;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  template <typename ActivePolicyT,
            typename BlockLevelSegmentedScanKernelT,
            typename WarpLevelSegmentedScanKernelT,
            typename ThreadLevelSegmentedScanKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t invoke_passes(
    BlockLevelSegmentedScanKernelT large_segmented_scan_kernel,
    WarpLevelSegmentedScanKernelT medium_segmented_scan_kernel,
    ThreadLevelSegmentedScanKernelT small_segmented_scan_kernel,
    ActivePolicyT policy = {})
  {
    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    policy.CheckLoadModifier();

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments per worker parameter must be positive");

    const auto [workers_per_block, block_size] = [](auto policy, worker selector) -> ::cuda::std::tuple<int, int> {
      switch (selector)
      {
        case worker::block: {
          const auto bw = policy.BlockWorkerSegmentedScan();
          return {bw.WorkersPerBlock(), bw.Config().BlockThreads()};
        }
        case worker::warp: {
          const auto ww = policy.WarpWorkerSegmentedScan();
          return {ww.WorkersPerBlock(), ww.Config().BlockThreads()};
        }
        case worker::thread: {
          const auto tw = policy.ThreadWorkerSegmentedScan();
          return {tw.WorkersPerBlock(), tw.Config().BlockThreads()};
        }
        default:
          _CCCL_UNREACHABLE();
      }
      _CCCL_UNREACHABLE();
    }(policy, worker_choice);
    const auto segments_per_block = num_segments_per_worker * workers_per_block;
    _CCCL_ASSERT(segments_per_block > 0, "Number of segments to be processed by block must be positive");

    static constexpr auto int32_max                       = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();
    static constexpr auto max_num_segments_per_invocation = static_cast<::cuda::std::int64_t>(int32_max);

    const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, max_num_segments_per_invocation);

    for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
    {
      const auto current_seg_offset          = invocation_index * max_num_segments_per_invocation;
      const auto next_seg_offset             = current_seg_offset + max_num_segments_per_invocation;
      const auto num_segments_per_invocation = ::cuda::std::min(next_seg_offset, num_segments) - current_seg_offset;

      _CCCL_ASSERT(num_segments_per_invocation <= max_num_segments_per_invocation,
                   "data loss during narrowing: num_segments_per_invocation exceeds int32_t range");

      const auto grid_size = ::cuda::ceil_div(static_cast<int>(num_segments_per_invocation), segments_per_block);

      auto launcher = launcher_factory(grid_size, block_size, 0, stream);

      // Cast is safe, since OffsetT is integral with sizeof(OffsetT) >= 4, and num_segments_per_invocation
      // fits in int32_t by construction
      const auto segment_count = static_cast<OffsetT>(num_segments_per_invocation);

      switch (worker_choice)
      {
        case worker::block:
          launcher.doit(
            large_segmented_scan_kernel,
            d_in,
            d_out,
            d_input_begin_offsets,
            d_input_end_offsets,
            d_output_begin_offsets,
            segment_count,
            scan_op,
            init_value,
            num_segments_per_worker);
          break;
        case worker::warp:
          launcher.doit(
            medium_segmented_scan_kernel,
            d_in,
            d_out,
            d_input_begin_offsets,
            d_input_end_offsets,
            d_output_begin_offsets,
            segment_count,
            scan_op,
            init_value,
            num_segments_per_worker);
          break;
        case worker::thread:
          launcher.doit(
            small_segmented_scan_kernel,
            d_in,
            d_out,
            d_input_begin_offsets,
            d_input_end_offsets,
            d_output_begin_offsets,
            segment_count,
            scan_op,
            init_value,
            num_segments_per_worker);
          break;
        default:
          _CCCL_UNREACHABLE();
      }

      cudaError_t error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        return error;
      }

      if (invocation_index + 1 < num_invocations)
      {
        d_input_begin_offsets += num_segments_per_invocation;
        d_input_end_offsets += num_segments_per_invocation;
        d_output_begin_offsets += num_segments_per_invocation;
      }

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        return error;
      }
    }

    return cudaSuccess;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    auto wrapped_policy = detail::segmented_scan::make_segmented_scan_policy_wrapper(policy);
    // Force kernel code-generation in all compiler passes
    return invoke_passes(
      kernel_source.segmented_scan_kernel(),
      kernel_source.warp_segmented_scan_kernel(),
      kernel_source.thread_segmented_scan_kernel(),
      wrapped_policy);
  }

  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorInputT input_begin_offsets,
    EndOffsetIteratorInputT input_end_offsets,
    BeginOffsetIteratorOutputT output_begin_offsets,
    ScanOpT scan_op,
    InitValueT init_value,
    int num_segments_per_worker,
    worker worker_choice,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    if (num_segments <= 0)
    {
      return cudaSuccess;
    }

    int ptx_version = 0;
    cudaError error = CubDebug(launcher_factory.PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    dispatch_segmented_scan dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      input_begin_offsets,
      input_end_offsets,
      output_begin_offsets,
      scan_op,
      init_value,
      num_segments_per_worker,
      worker_choice,
      stream,
      ptx_version,
      kernel_source,
      launcher_factory};

    error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
    if (cudaSuccess != error)
    {
      return error;
    }

    return error;
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
