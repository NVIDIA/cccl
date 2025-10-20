// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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

#include <cub/agent/agent_segmented_scan.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/segmented_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/thread/thread_operators.cuh>
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
struct DeviceSegmentedScanKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    SegmentedScanKernel,
    DeviceSegmentedScanKernel<
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
      EnforceInclusive == ForceInclusive::Yes>)

  CUB_RUNTIME_FUNCTION static constexpr size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};

} // namespace detail::segmented_scan

// TODO: define struct DispatchSegmentedScan

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
  typename _OffsetT               = typename detail::
    common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>,
  typename PolicyHub = detail::segmented_scan::
    policy_hub<detail::it_value_t<InputIteratorT>, detail::it_value_t<OutputIteratorT>, AccumT, _OffsetT, ScanOpT>,
  typename KernelSource = detail::segmented_scan::DeviceSegmentedScanKernelSource<
    typename PolicyHub::MaxPolicy,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorInputT,
    EndOffsetIteratorInputT,
    BeginOffsetIteratorOutputT,
    _OffsetT,
    ScanOpT,
    InitValueT,
    AccumT,
    EnforceInclusive>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchSegmentedScan
{
  using OffsetT = _OffsetT;

  static_assert(::cuda::std::is_integral_v<OffsetT> && sizeof(OffsetT) >= 4 && sizeof(OffsetT) <= 8,
                "DispatchSegmentedScan only supports integral offset types of 4- or 8-bytes");

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

  BeginOffsetIteratorInputT d_input_begin_offsets;

  EndOffsetIteratorInputT d_input_end_offsets;

  BeginOffsetIteratorOutputT d_output_begin_offsets;

  /// Binary scan functor
  ScanOpT scan_op;

  /// Initial value to seed the exclusive scan
  InitValueT init_value;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  // Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_segments(num_segments)
      , d_input_begin_offsets(d_in_begin_offsets)
      , d_input_end_offsets(d_in_end_offsets)
      , d_output_begin_offsets(d_out_begin_offsets)
      , scan_op(scan_op)
      , init_value(init_value)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  template <typename ActivePolicyT, typename SegmentedScanKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_HOST _CCCL_FORCEINLINE cudaError_t
  InvokePasses(SegmentedScanKernelT segmented_scan_kernel, ActivePolicyT policy = {})
  {
    // `LOAD_LDG` makes in-place execution UB and doesn't lead to better
    // performance.
    policy.CheckLoadModifier();

    cudaError error = cudaSuccess;
    do
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        return cudaSuccess;
      }

      const auto int32_max                       = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();
      const auto num_segments_per_invocation     = static_cast<::cuda::std::int64_t>(int32_max);
      const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, num_segments_per_invocation);

      for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
      {
        const auto current_seg_offset = invocation_index * num_segments_per_invocation;
        const auto num_current_segments =
          ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);

        // Invoke DeviceSegmentedScanKernel
        launcher_factory(
          static_cast<::cuda::std::uint32_t>(num_current_segments), policy.SegmentedScan().BlockThreads(), 0, stream)
          .doit(segmented_scan_kernel,
                d_in,
                d_out,
                d_input_begin_offsets,
                d_input_end_offsets,
                d_output_begin_offsets,
                static_cast<OffsetT>(num_current_segments),
                scan_op,
                init_value);

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        if (invocation_index + 1 < num_invocations)
        {
          d_input_begin_offsets += num_current_segments;
          d_input_end_offsets += num_current_segments;
          d_output_begin_offsets += num_current_segments;
        }

        // Sync the stream if specified to flush runtime errors
        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);
    return error;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    auto wrapped_policy = detail::segmented_scan::MakeSegmentedScanPolicyWrapper(policy);
    // Force kernel code-generation in all compiler passes
    return InvokePasses(kernel_source.SegmentedScanKernel(), wrapped_policy);
  }

  /**
   * @brief Internal dispatch routine
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no
   *   work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Iterator to the input sequence of data items
   *
   * @param[out] d_out
   *   Iterator to the output sequence of data items
   *
   * @param[in] num_segments
   *   Total number of segments that comprise the input data
   *
   * @param[in] input_begin_offsets
   *   Random-access iterator to the offsets for beginnings of segments in
   *   the input sequence
   *
   * @param[in] input_end_offsets
   *   Random-access iterator to the offsets for endings of segments in
   *   the input sequence
   *
   * @param[in] output_begin_offsets
   *   Random-access iterator to the offsets for beginnings of segments in
   *   the output sequence
   *
   * @param[in] scan_op
   *   Binary scan functor
   *
   * @param[in] init_value
   *   Initial value to seed the exclusive scan
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   *
   * @param[in] kernel_source
   *   Object specifying implementation kernels
   *
   * @param[in] launcher_factory
   *   Object to execute implementation kernels on the given stream
   *
   * @param[in] max_policy
   *   Struct encoding chain of algorithm tuning policies
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
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
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    if (num_segments <= 0)
    {
      return cudaSuccess;
    }

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchSegmentedScan dispatch(
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
        stream,
        ptx_version,
        kernel_source,
        launcher_factory);

      // Dispatch to chained policy
      error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }
};

CUB_NAMESPACE_END
