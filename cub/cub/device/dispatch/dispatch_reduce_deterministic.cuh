// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file This file device-wide, parallel operations for
 *       computing a reduction across a sequence of data items residing within
 *       device-accessible memory. Current reduction operator supported is ::cuda::std::plus
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

#include <cub/agent/agent_reduce.cuh>
#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::rfa
{
template <typename Invocable, typename InputT>
using transformed_input_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<Invocable, InputT>>;

template <typename InitT, typename InputIteratorT, typename TransformOpT>
using accum_t =
  ::cuda::std::__accumulator_t<::cuda::std::plus<>, InitT, transformed_input_t<TransformOpT, it_value_t<InputIteratorT>>>;

template <typename FloatType                                                              = float,
          typename ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<FloatType>>* = nullptr>
struct deterministic_sum_t
{
  using DeterministicAcc = ReproducibleFloatingAccumulator<FloatType>;

  _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc acc, FloatType f)
  {
    acc += f;
    return acc;
  }

  _CCCL_DEVICE DeterministicAcc operator()(FloatType f, DeterministicAcc acc)
  {
    return this->operator()(acc, f);
  }

  _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc lhs, DeterministicAcc rhs)
  {
    DeterministicAcc rtn = lhs;
    rtn += rhs;
    return rtn;
  }

  _CCCL_DEVICE FloatType operator()(FloatType lhs, FloatType rhs)
  {
    return lhs + rhs;
  }
};

/******************************************************************************
 * Single-problem dispatch
 *****************************************************************************/
/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction in deterministic fashion
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
 * @tparam InitT
 *   Initial value type
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename InitT,
          typename TransformOpT = ::cuda::std::identity,
          typename AccumT       = accum_t<InitT, InputIteratorT, TransformOpT>,
          typename PolicyHub    = policy_hub<AccumT, OffsetT, ::cuda::std::plus<>>>
struct dispatch_t
{
  using deterministic_add_t = deterministic_sum_t<AccumT>;
  using reduction_op_t      = deterministic_add_t;

  using deterministic_accum_t = typename deterministic_add_t::DeterministicAcc;
  using input_unwrapped_it_t  = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;

  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Unwrapped Pointer to the input sequence of data items
  input_unwrapped_it_t d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Binary reduction functor
  reduction_op_t reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  TransformOpT transform_op = {};

  //---------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke a single block block to reduce in-core deterministically
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeterministicDeviceReduceSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeterministicDeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel)
  {
    // Return if the caller is simply requesting the size of the storage
    // allocation
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }
// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
            (long long) stream,
            ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
#endif
    // Invoke single_reduce_sweep_kernel
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
      .doit(single_tile_kernel, d_in, d_out, static_cast<int>(num_items), reduction_op, init, transform_op);
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
   * @brief Invoke two-passes to reduce deteerministically
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam ReduceKernelT
   *   Function type of cub::DeterministicDeviceReduceKernel
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeterministicDeviceReduceSingleTileKernel
   *
   * @param[in] reduce_kernel
   *   Kernel function pointer to parameterization of cub::DeterministicDeviceReduceKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeterministicDeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename ReduceKernelT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(ReduceKernelT reduce_kernel, SingleTileKernelT single_tile_kernel)
  {
    const auto tile_size = ActivePolicyT::ReducePolicy::BLOCK_THREADS * ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD;
    // Get device ordinal
    int device_ordinal;
    if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
    {
      return error;
    }

    int sm_count;
    if (const auto error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
    {
      return error;
    }

    KernelConfig reduce_config;
    if (const auto error = CubDebug(reduce_config.Init<typename ActivePolicyT::ReducePolicy>(reduce_kernel)))
    {
      return error;
    }

    const int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;
    const int max_blocks              = reduce_device_occupancy * CUB_SUBSCRIPTION_FACTOR(0);

    const int num_items_per_chunk = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();
    const int num_chunks          = static_cast<int>(::cuda::ceil_div(num_items, num_items_per_chunk));

    const int chunk_tile_grid_size = ::cuda::ceil_div(num_items_per_chunk, tile_size);
    const int chunk_grid_size      = ::cuda::std::min(max_blocks, chunk_tile_grid_size);

    const int partial_chunk_size        = num_items % num_items_per_chunk;
    const bool has_partial_chunk        = partial_chunk_size != 0;
    const int last_chunk_tile_grid_size = ::cuda::ceil_div(partial_chunk_size, tile_size);

    const int last_chunk_grid_size = ::cuda::std::min(max_blocks, last_chunk_tile_grid_size);

    const int reduce_grid_size = chunk_grid_size * (num_chunks - 1) + last_chunk_grid_size;

    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {
      reduce_grid_size * sizeof(deterministic_accum_t) // bytes needed for privatized block
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
    deterministic_accum_t* d_block_reductions = (deterministic_accum_t*) allocations[0];

    auto d_chunk_block_reductions = d_block_reductions;
    for (int chunk_index = 0; chunk_index < num_chunks; chunk_index++)
    {
      const int num_current_items =
        ((chunk_index + 1 == num_chunks) && has_partial_chunk) ? partial_chunk_size : num_items_per_chunk;

      const auto current_grid_size =
        static_cast<int>(num_current_items == num_items_per_chunk ? chunk_grid_size : last_chunk_grid_size);

      // Log device_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeterministicDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items "
              "per thread, %d SM occupancy\n",
              current_grid_size,
              ActivePolicyT::ReducePolicy::BLOCK_THREADS,
              (long long) stream,
              ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD,
              reduce_config.sm_occupancy);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
        current_grid_size, ActivePolicyT::ReducePolicy::BLOCK_THREADS, 0, stream)
        .doit(reduce_kernel,
              d_in,
              d_chunk_block_reductions,
              num_current_items,
              reduction_op,
              transform_op,
              current_grid_size);

      // Check for failure to launch
      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }

      if (chunk_index + 1 < num_chunks)
      {
        d_in += num_current_items;
        d_chunk_block_reductions += current_grid_size;
      }

      // Sync the stream if specified to flush runtime errors
      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }
    }
// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
            (long long) stream,
            ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

    // Invoke DeterministicDeviceReduceSingleTileKernel
    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
      .doit(single_tile_kernel,
            d_block_reductions,
            d_out,
            reduce_grid_size, // triple_chevron is not type safe, make sure to use int
            reduction_op,
            init,
            ::cuda::std::identity{});

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

  /// Invocation Deterministic
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using SingleTilePolicyT = typename ActivePolicyT::SingleTilePolicy;
    using MaxPolicyT        = typename PolicyHub::MaxPolicy;

    // Force kernel code-generation in all compiler passes
    if (num_items <= (SingleTilePolicyT::BLOCK_THREADS * SingleTilePolicyT::ITEMS_PER_THREAD))
    {
      return InvokeSingleTile<ActivePolicyT>(
        detail::reduce::DeterministicDeviceReduceSingleTileKernel<
          MaxPolicyT,
          input_unwrapped_it_t,
          OutputIteratorT,
          reduction_op_t,
          InitT,
          deterministic_accum_t,
          TransformOpT>);
    }
    else
    {
      return InvokePasses<ActivePolicyT>(
        detail::reduce::DeterministicDeviceReduceKernel<MaxPolicyT,
                                                        input_unwrapped_it_t,
                                                        reduction_op_t,
                                                        deterministic_accum_t,
                                                        TransformOpT>,
        detail::reduce::DeterministicDeviceReduceSingleTileKernel<
          MaxPolicyT,
          deterministic_accum_t*,
          OutputIteratorT,
          reduction_op_t,
          InitT,
          deterministic_accum_t>);
    }
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide deterministic reduction
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
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    InitT init                = {},
    cudaStream_t stream       = {},
    TransformOpT transform_op = {})
  {
    // Get PTX version
    int ptx_version = 0;
    if (const auto error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    input_unwrapped_it_t d_in_unwrapped = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);

    // Create dispatch functor
    dispatch_t dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_in_unwrapped,
      d_out,
      num_items,
      deterministic_add_t{},
      init,
      stream,
      ptx_version,
      transform_op};

    // Dispatch to chained policy
    return CubDebug(PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch));
  }
};
} // namespace detail::rfa
CUB_NAMESPACE_END
