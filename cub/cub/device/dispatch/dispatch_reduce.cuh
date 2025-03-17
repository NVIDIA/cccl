/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceReduce provides device-wide, parallel operations for
 *       computing a reduction across a sequence of data items residing within
 *       device-accessible memory.
 */

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
#include <cub/detail/type_traits.cuh> // for cub::detail::invoke_result_t
#include <cub/device/dispatch/kernels/reduce.cuh>
#include <cub/device/dispatch/kernels/segmented_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh> // for cub::detail::non_void_value_t, cub::detail::value_t

#include <cuda/std/functional>

#include <stdio.h>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{

template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT,
          typename TransformOpT>
struct DeviceReduceKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    SingleTileKernel,
    DeviceReduceSingleTileKernel<MaxPolicyT,
                                 InputIteratorT,
                                 OutputIteratorT,
                                 OffsetT,
                                 ReductionOpT,
                                 InitT,
                                 AccumT,
                                 TransformOpT>)

  CUB_DEFINE_KERNEL_GETTER(ReductionKernel,
                           DeviceReduceKernel<MaxPolicyT, InputIteratorT, OffsetT, ReductionOpT, AccumT, TransformOpT>)

  CUB_DEFINE_KERNEL_GETTER(
    SingleTileSecondKernel,
    DeviceReduceSingleTileKernel<MaxPolicyT,
                                 AccumT*,
                                 OutputIteratorT,
                                 int, // Always used with int offsets
                                 ReductionOpT,
                                 InitT,
                                 AccumT>)

  CUB_RUNTIME_FUNCTION static constexpr size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};
} // namespace detail::reduce

/******************************************************************************
 * Single-problem dispatch
 *****************************************************************************/

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
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT  = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOpT, cub::detail::it_value_t<InputIteratorT>, InitT>,
          typename TransformOpT = ::cuda::std::__identity,
          typename PolicyHub    = detail::reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
          typename KernelSource = detail::reduce::DeviceReduceKernelSource<
            typename PolicyHub::MaxPolicy,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT,
            InitT,
            AccumT,
            TransformOpT>,
          typename KernelLauncherFactory = detail::TripleChevronFactory>
struct DispatchReduce
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

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  TransformOpT transform_op;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream,
    int ptx_version,
    TransformOpT transform_op              = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
      , transform_op(transform_op)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //---------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke a single block block to reduce in-core
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceReduceSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel, ActivePolicyT policy = {})
  {
    cudaError error = cudaSuccess;
    do
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        break;
      }

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              policy.SingleTile().BlockThreads(),
              (long long) stream,
              policy.SingleTile().ItemsPerThread());
#endif // CUB_DEBUG_LOG

      // Invoke single_reduce_sweep_kernel
      launcher_factory(1, policy.SingleTile().BlockThreads(), 0, stream)
        .doit(single_tile_kernel, d_in, d_out, num_items, reduction_op, init, transform_op);

      // Check for failure to launch
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
    } while (0);

    return error;
  }

  //---------------------------------------------------------------------------
  // Normal problem size invocation (two-pass)
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke two-passes to reduce
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam ReduceKernelT
   *   Function type of cub::DeviceReduceKernel
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeviceReduceSingleTileKernel
   *
   * @param[in] reduce_kernel
   *   Kernel function pointer to parameterization of cub::DeviceReduceKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename ReduceKernelT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(ReduceKernelT reduce_kernel, SingleTileKernelT single_tile_kernel, ActivePolicyT active_policy = {})
  {
    cudaError error = cudaSuccess;
    do
    {
      // Get SM count
      int sm_count;
      error = CubDebug(launcher_factory.MultiProcessorCount(sm_count));
      // error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Init regular kernel configuration
      detail::KernelConfig reduce_config;
      error = CubDebug(reduce_config.Init(reduce_kernel, active_policy.Reduce(), launcher_factory));
      if (cudaSuccess != error)
      {
        break;
      }

      int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;

      // Even-share work distribution
      int max_blocks = reduce_device_occupancy * subscription_factor;
      GridEvenShare<OffsetT> even_share;
      even_share.DispatchInit(num_items, max_blocks, reduce_config.tile_size);

      // Temporary storage allocation requirements
      void* allocations[1]       = {};
      size_t allocation_sizes[1] = {
        max_blocks * kernel_source.AccumSize() // bytes needed for privatized block
                                               // reductions
      };

      // Alias the temporary allocations from the single storage blob (or
      // compute the necessary size of the blob)
      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        return cudaSuccess;
      }

      // Alias the allocation for the privatized per-block reductions
      AccumT* d_block_reductions = static_cast<AccumT*>(allocations[0]);

      // Get grid size for device_reduce_sweep_kernel
      int reduce_grid_size = even_share.grid_size;

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking DeviceReduceKernel<<<%lu, %d, 0, %lld>>>(), %d items "
              "per thread, %d SM occupancy\n",
              (unsigned long) reduce_grid_size,
              active_policy.Reduce().BlockThreads(),
              (long long) stream,
              active_policy.Reduce().ItemsPerThread(),
              reduce_config.sm_occupancy);
#endif // CUB_DEBUG_LOG

      // Invoke DeviceReduceKernel
      launcher_factory(reduce_grid_size, active_policy.Reduce().BlockThreads(), 0, stream)
        .doit(reduce_kernel, d_in, d_block_reductions, num_items, even_share, reduction_op, transform_op);

      // Check for failure to launch
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

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking DeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              active_policy.SingleTile().BlockThreads(),
              (long long) stream,
              active_policy.SingleTile().ItemsPerThread());
#endif // CUB_DEBUG_LOG

      // Invoke DeviceReduceSingleTileKernel
      launcher_factory(1, active_policy.SingleTile().BlockThreads(), 0, stream)
        .doit(single_tile_kernel,
              d_block_reductions,
              d_out,
              reduce_grid_size,
              reduction_op,
              init,
              ::cuda::std::__identity{});

      // Check for failure to launch
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
    } while (0);

    return error;
  }

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    auto wrapped_policy = detail::reduce::MakeReducePolicyWrapper(active_policy);
    if (num_items <= static_cast<OffsetT>(
          wrapped_policy.SingleTile().BlockThreads() * wrapped_policy.SingleTile().ItemsPerThread()))
    {
      // Small, single tile size
      return InvokeSingleTile(kernel_source.SingleTileKernel(), wrapped_policy);
    }
    else
    {
      // Regular size
      return InvokePasses(kernel_source.ReductionKernel(), kernel_source.SingleTileSecondKernel(), wrapped_policy);
    }
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
   * @param[in] num_items
   *   Total number of input items (i.e., length of `d_in`)
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
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream,
    TransformOpT transform_op              = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
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
      DispatchReduce dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        reduction_op,
        init,
        stream,
        ptx_version,
        transform_op,
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

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide transform reduce
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam TransformOpT
 *   Unary transform functor type having member
 *   `auto operator()(const T &a)`
 *
 * @tparam InitT
 *   Initial value type
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename OffsetT,
  typename ReductionOpT,
  typename TransformOpT,
  typename InitT,
  typename AccumT =
    ::cuda::std::__accumulator_t<ReductionOpT,
                                 cub::detail::invoke_result_t<TransformOpT, cub::detail::it_value_t<InputIteratorT>>,
                                 InitT>,
  typename PolicyHub    = detail::reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
  typename KernelSource = detail::reduce::DeviceReduceKernelSource<
    typename PolicyHub::MaxPolicy,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ReductionOpT,
    InitT,
    AccumT,
    TransformOpT>,
  typename KernelLauncherFactory = detail::TripleChevronFactory>
using DispatchTransformReduce =
  DispatchReduce<InputIteratorT,
                 OutputIteratorT,
                 OffsetT,
                 ReductionOpT,
                 InitT,
                 AccumT,
                 TransformOpT,
                 PolicyHub,
                 KernelSource,
                 KernelLauncherFactory>;

/******************************************************************************
 * Segmented dispatch
 *****************************************************************************/

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
          typename KernelLauncherFactory = detail::TripleChevronFactory>
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

  /// The number of segments that comprise the sorting data
  int num_segments;

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
    int num_segments,
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
   *   Kernel function pointer to parameterization of
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

      // Init kernel configuration
      [[maybe_unused]] detail::KernelConfig segmented_reduce_config;
      error =
        CubDebug(segmented_reduce_config.Init(segmented_reduce_kernel, policy.SegmentedReduce(), launcher_factory));
      if (cudaSuccess != error)
      {
        break;
      }

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
      _CubLog("Invoking SegmentedDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), "
              "%d items per thread, %d SM occupancy\n",
              num_segments,
              policy.SegmentedReduce().BlockThreads(),
              (long long) stream,
              policy.SegmentedReduce().ItemsPerThread(),
              segmented_reduce_config.sm_occupancy);
#endif // CUB_DEBUG_LOG

      // Invoke DeviceReduceKernel
      launcher_factory(num_segments, policy.SegmentedReduce().BlockThreads(), 0, stream)
        .doit(segmented_reduce_kernel, d_in, d_out, d_begin_offsets, d_end_offsets, num_segments, reduction_op, init);

      // Check for failure to launch
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
    int num_segments,
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
