// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh> // for cub::detail::non_void_value_t, cub::detail::it_value_t

#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/cstdint>

// TODO(bgruber): included to not break users when moving DeviceSegmentedReduce to its own file. Remove in CCCL 4.0.
#include <cub/device/dispatch/dispatch_fixed_size_segmented_reduce.cuh>
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

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
          typename TransformOpT = ::cuda::std::identity,
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
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
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
    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
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
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
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
    // Get SM count
    int sm_count;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
    {
      return error;
    }

    // Init regular kernel configuration
    detail::KernelConfig reduce_config;
    if (const auto error = CubDebug(reduce_config.Init(reduce_kernel, active_policy.Reduce(), launcher_factory)))
    {
      return error;
    }

    int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;

    // Even-share work distribution
    int max_blocks = reduce_device_occupancy * detail::subscription_factor;
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
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
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
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
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
      .doit(
        single_tile_kernel, d_block_reductions, d_out, reduce_grid_size, reduction_op, init, ::cuda::std::identity{});

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    return CubDebug(detail::DebugSyncStream(stream));
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
    // Get PTX version
    int ptx_version = 0;
    if (const auto error = CubDebug(launcher_factory.PtxVersion(ptx_version)))
    {
      return error;
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

    // Ignore Wmaybe-uninitialized to work around a GCC 13 issue:
    // https://github.com/NVIDIA/cccl/issues/4053
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wmaybe-uninitialized")
    // Dispatch to chained policy
    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
    _CCCL_DIAG_POP
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
                                 ::cuda::std::invoke_result_t<TransformOpT, ::cuda::std::iter_value_t<InputIteratorT>>,
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
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
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

CUB_NAMESPACE_END
