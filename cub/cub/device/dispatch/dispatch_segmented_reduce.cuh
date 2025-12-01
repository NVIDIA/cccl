// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cub/device/dispatch/kernels/kernel_segmented_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh> // for cub::detail::non_void_value_t, cub::detail::it_value_t

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
struct DeviceSegmentedReduceKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    SegmentedReduceKernel,
    DeviceSegmentedReduceKernel<
      MaxPolicyT,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      OffsetT,
      ReductionOpT,
      InitT,
      AccumT>)

  CUB_RUNTIME_FUNCTION static constexpr ::cuda::std::size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};
} // namespace detail::reduce

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets
 *   @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets
 *   @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   value type
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT  = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOpT, cub::detail::it_value_t<InputIteratorT>, InitT>,
          typename PolicyHub    = detail::reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
          typename KernelSource = detail::reduce::DeviceSegmentedReduceKernelSource<
            typename PolicyHub::MaxPolicy,
            InputIteratorT,
            OutputIteratorT,
            BeginOffsetIteratorT,
            EndOffsetIteratorT,
            OffsetT,
            ReductionOpT,
            InitT,
            AccumT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchSegmentedReduce
{
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// The number of segments that comprise the segmented reduction data
  ::cuda::std::int64_t num_segments;

  /// Random-access input iterator to the sequence of beginning offsets of
  /// length `num_segments`, such that `d_begin_offsets[i]` is the first
  /// element of the *i*<sup>th</sup> data segment in `d_keys_*` and
  /// `d_values_*`
  BeginOffsetIteratorT d_begin_offsets;

  /// Random-access input iterator to the sequence of ending offsets of length
  /// `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
  /// the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
  /// If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
  /// considered empty.
  EndOffsetIteratorT d_end_offsets;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  // Source getter
  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSegmentedReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invocation
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam DeviceSegmentedReduceKernelT
   *   Function type of cub::DeviceSegmentedReduceKernel
   *
   * @param[in] segmented_reduce_kernel
   *   Kernel function pointer to instantiation of
   *   cub::DeviceSegmentedReduceKernel
   */
  template <typename ActivePolicyT, typename DeviceSegmentedReduceKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(DeviceSegmentedReduceKernelT segmented_reduce_kernel, ActivePolicyT policy = {})
  {
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

      // Init kernel configuration (computes kernel occupancy)
      // maybe only used inside CUB_DEBUG_LOG code sections
      [[maybe_unused]] detail::KernelConfig segmented_reduce_config;
      error =
        CubDebug(segmented_reduce_config.Init(segmented_reduce_kernel, policy.SegmentedReduce(), launcher_factory));
      if (cudaSuccess != error)
      {
        break;
      }

      const auto num_segments_per_invocation =
        static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<::cuda::std::int32_t>::max());
      const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, num_segments_per_invocation);

      for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
      {
        const auto current_seg_offset = invocation_index * num_segments_per_invocation;
        const auto num_current_segments =
          ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
        _CubLog("Invoking SegmentedDeviceReduceKernel<<<%ld, %d, 0, %lld>>>(), "
                "%d items per thread, %d SM occupancy\n",
                num_current_segments,
                policy.SegmentedReduce().BlockThreads(),
                (long long) stream,
                policy.SegmentedReduce().ItemsPerThread(),
                segmented_reduce_config.sm_occupancy);
#endif // CUB_DEBUG_LOG

        // Invoke DeviceSegmentedReduceKernel
        launcher_factory(
          static_cast<::cuda::std::uint32_t>(num_current_segments), policy.SegmentedReduce().BlockThreads(), 0, stream)
          .doit(segmented_reduce_kernel, d_in, d_out, d_begin_offsets, d_end_offsets, reduction_op, init);

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        if (invocation_index + 1 < num_invocations)
        {
          d_out += num_current_segments;
          d_begin_offsets += num_current_segments;
          d_end_offsets += num_current_segments;
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
    auto wrapped_policy = detail::reduce::MakeReducePolicyWrapper(policy);
    // Force kernel code-generation in all compiler passes
    return InvokePasses(kernel_source.SegmentedReduceKernel(), wrapped_policy);
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out
   *   Pointer to the output aggregate
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
   *   considered empty.
   *
   * @param[in] reduction_op
   *   Binary reduction functor
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitT init,
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
      DispatchSegmentedReduce dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        reduction_op,
        init,
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
