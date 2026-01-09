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
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN
namespace detail::reduce
{
// @brief Functor to generate a key-value pair from an index and value
template <typename Iterator, typename OutputValueT>
struct generate_idx_value
{
private:
  Iterator it;
  int segment_size;

public:
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE generate_idx_value(Iterator it, int segment_size)
      : it(it)
      , segment_size(segment_size)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(::cuda::std::int64_t idx) const
  {
    return ::cuda::std::pair<int, OutputValueT>(static_cast<int>(idx % segment_size), it[idx]);
  }
};

template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT>
struct DeviceFixedSizeSegmentedReduceKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    FixedSizeSegmentedReduceKernel,
    DeviceFixedSizeSegmentedReduceKernel<MaxPolicyT, InputIteratorT, OutputIteratorT, OffsetT, ReductionOpT, InitT, AccumT>)

  CUB_RUNTIME_FUNCTION static constexpr ::cuda::std::size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT       = ::cuda::std::__accumulator_t<ReductionOpT, it_value_t<InputIteratorT>, InitT>,
          typename PolicyHub    = fixed_size_segmented_reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
          typename KernelSource = DeviceFixedSizeSegmentedReduceKernelSource<
            typename PolicyHub::MaxPolicy,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT,
            InitT,
            AccumT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct DispatchFixedSizeSegmentedReduce
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

  /// The fixed segment size for each segment
  OffsetT segment_size;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchFixedSizeSegmentedReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    OffsetT segment_size,
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
      , segment_size(segment_size)
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
   * @tparam DeviceFixedSizeSegmentedReduceKernelT
   *   Function type of cub::DeviceFixedSizeSegmentedReduceKernel
   *
   * @param[in] fixed_size_segmented_reduce_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceFixedSizeSegmentedReduceKernel
   */
  template <typename ActivePolicyT, typename DeviceFixedSizeSegmentedReduceKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(DeviceFixedSizeSegmentedReduceKernelT fixed_size_segmented_reduce_kernel)
  {
    constexpr auto small_items_per_tile  = ActivePolicyT::SmallReducePolicy::ITEMS_PER_TILE;
    constexpr auto medium_items_per_tile = ActivePolicyT::MediumReducePolicy::ITEMS_PER_TILE;

    static_assert((small_items_per_tile < medium_items_per_tile),
                  "small items per tile must be less than medium items per tile");

    // Return if the caller is simply requesting the size of the storage
    // allocation
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    // assume large segment size problem
    int segments_per_block = 1;

    if (segment_size <= small_items_per_tile) // small segment size problem
    {
      segments_per_block = ActivePolicyT::SmallReducePolicy::SEGMENTS_PER_BLOCK;
    }
    else if (segment_size <= medium_items_per_tile) // medium segment size problem
    {
      segments_per_block = ActivePolicyT::MediumReducePolicy::SEGMENTS_PER_BLOCK;
    }

    const auto num_segments_per_invocation =
      static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<::cuda::std::int32_t>::max());

    const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, num_segments_per_invocation);

    cudaError error = cudaSuccess;
    for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
    {
      const auto current_seg_offset = invocation_index * num_segments_per_invocation;

      const auto num_current_segments =
        ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);

      const auto num_current_blocks = ::cuda::ceil_div(num_current_segments, segments_per_block);

      launcher_factory(
        static_cast<::cuda::std::int32_t>(num_current_blocks), ActivePolicyT::ReducePolicy::BLOCK_THREADS, 0, stream)
        .doit(fixed_size_segmented_reduce_kernel,
              d_in,
              d_out,
              segment_size,
              static_cast<::cuda::std::int32_t>(num_current_segments),
              reduction_op,
              init);

      d_in += num_segments_per_invocation * segment_size;
      d_out += num_segments_per_invocation;

      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    }
    return error;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    return InvokePasses<ActivePolicyT>(kernel_source.FixedSizeSegmentedReduceKernel());
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide segmented reduction
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
   *   Pointer to the output aggregates
   *
   * @param[in] num_segments
   *   The number of segments that comprise the segmented reduction data
   *
   * @param[in] segment_size
   *   The fixed segment size for each segment
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
    OffsetT segment_size,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    if (num_segments <= 0)
    {
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
      }
      return cudaSuccess;
    }

    // Get PTX version
    int ptx_version = 0;
    cudaError error = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    // Create dispatch functor
    DispatchFixedSizeSegmentedReduce dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      segment_size,
      reduction_op,
      init,
      stream,
      ptx_version,
      kernel_source,
      launcher_factory);

    // Dispatch to chained policy
    error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
    return error;
  }
};
} // namespace detail::reduce
CUB_NAMESPACE_END
