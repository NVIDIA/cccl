// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! This file device-wide, parallel operations for computing a reduction across a sequence of data items residing within
//! device-accessible memory. Current reduction operator supported is ::cuda::std::plus

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
#include <cub/device/dispatch/kernels/kernel_reduce_deterministic.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_deterministic.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

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

template <typename FloatType = float, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<FloatType>>* = nullptr>
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

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename DeterministicAccumT,
          typename TransformOpT,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t invoke_single_tile(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitT init,
  cudaStream_t stream,
  TransformOpT transform_op,
  rfa_policy active_policy,
  KernelLauncherFactory launcher_factory)
{
  // Return if the caller is simply requesting the size of the storage allocation
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
          "%d items per thread\n",
          active_policy.single_tile.block_threads,
          (long long) stream,
          active_policy.single_tile.items_per_thread);
#endif // CUB_DEBUG_LOG

  // Invoke single_reduce_sweep_kernel
  if (const auto error = CubDebug(
        launcher_factory(1, active_policy.single_tile.block_threads, 0, stream)
          .doit(detail::reduce::DeterministicDeviceReduceSingleTileKernel<
                  PolicySelector,
                  InputIteratorT,
                  OutputIteratorT,
                  ReductionOpT,
                  InitT,
                  DeterministicAccumT,
                  TransformOpT>,
                d_in,
                d_out,
                static_cast<int>(num_items),
                reduction_op,
                init,
                transform_op)))
  {
    return error;
  }

  // Check for failure to launch
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Sync the stream if specified to flush runtime errors
  return CubDebug(detail::DebugSyncStream(stream));
}

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename DeterministicAccumT,
          typename TransformOpT,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t invoke_passes(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitT init,
  cudaStream_t stream,
  TransformOpT transform_op,
  rfa_policy active_policy,
  KernelLauncherFactory launcher_factory)
{
  int sm_count;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  KernelConfig reduce_config;
  if (const auto error = CubDebug(reduce_config.__init(
        detail::reduce::
          DeterministicDeviceReduceKernel<PolicySelector, InputIteratorT, ReductionOpT, DeterministicAccumT, TransformOpT>,
        active_policy.reduce)))
  {
    return error;
  }

  const int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;
  const int max_blocks              = reduce_device_occupancy * detail::subscription_factor;

  const int num_items_per_chunk = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();
  const int num_chunks          = static_cast<int>(::cuda::ceil_div(num_items, num_items_per_chunk));

  const int chunk_tile_grid_size = ::cuda::ceil_div(num_items_per_chunk, reduce_config.tile_size);
  const int chunk_grid_size      = ::cuda::std::min(max_blocks, chunk_tile_grid_size);

  const int partial_chunk_size        = num_items % num_items_per_chunk;
  const bool has_partial_chunk        = partial_chunk_size != 0;
  const int last_chunk_tile_grid_size = ::cuda::ceil_div(partial_chunk_size, reduce_config.tile_size);

  const int last_chunk_grid_size = ::cuda::std::min(max_blocks, last_chunk_tile_grid_size);

  const int reduce_grid_size = chunk_grid_size * (num_chunks - 1) + last_chunk_grid_size;

  // Temporary storage allocation requirements
  void* allocations[1]       = {};
  size_t allocation_sizes[1] = {
    reduce_grid_size * sizeof(DeterministicAccumT) // bytes needed for privatized block reductions
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
    // Return if the caller is simply requesting the size of the storage allocation
    return cudaSuccess;
  }

  // Alias the allocation for the privatized per-block reductions
  DeterministicAccumT* d_block_reductions = static_cast<DeterministicAccumT*>(allocations[0]);

  auto d_chunk_block_reductions = d_block_reductions;
  for (int chunk_index = 0; chunk_index < num_chunks; chunk_index++)
  {
    const int num_current_items =
      ((chunk_index + 1 == num_chunks) && has_partial_chunk) ? partial_chunk_size : num_items_per_chunk;

    const auto current_grid_size =
      static_cast<int>(num_current_items == num_items_per_chunk ? chunk_grid_size : last_chunk_grid_size);

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeterministicDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items "
            "per thread, %d SM occupancy\n",
            current_grid_size,
            active_policy.reduce.block_threads,
            (long long) stream,
            active_policy.reduce.items_per_thread,
            reduce_config.sm_occupancy);
#endif // CUB_DEBUG_LOG

    if (const auto error = CubDebug(
          launcher_factory(current_grid_size, active_policy.reduce.block_threads, 0, stream)
            .doit(detail::reduce::DeterministicDeviceReduceKernel<PolicySelector,
                                                                  InputIteratorT,
                                                                  ReductionOpT,
                                                                  DeterministicAccumT,
                                                                  TransformOpT>,
                  d_in,
                  d_chunk_block_reductions,
                  num_current_items,
                  reduction_op,
                  transform_op,
                  current_grid_size)))
    {
      return error;
    }

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
#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
          "%d items per thread\n",
          active_policy.single_tile.block_threads,
          (long long) stream,
          active_policy.single_tile.items_per_thread);
#endif // CUB_DEBUG_LOG

  // Invoke DeterministicDeviceReduceSingleTileKernel
  if (const auto error = CubDebug(
        launcher_factory(1, active_policy.single_tile.block_threads, 0, stream)
          .doit(detail::reduce::DeterministicDeviceReduceSingleTileKernel<
                  PolicySelector,
                  DeterministicAccumT*,
                  OutputIteratorT,
                  ReductionOpT,
                  InitT,
                  DeterministicAccumT>,
                d_block_reductions,
                d_out,
                reduce_grid_size,
                reduction_op,
                init,
                ::cuda::std::identity{})))
  {
    return error;
  }

  // Check for failure to launch
  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Sync the stream if specified to flush runtime errors
  return CubDebug(detail::DebugSyncStream(stream));
}

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename InitT,
          typename TransformOpT          = ::cuda::std::identity,
          typename AccumT                = accum_t<InitT, InputIteratorT, TransformOpT>,
          typename PolicySelector        = policy_selector_from_types<AccumT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  InitT init                             = {},
  cudaStream_t stream                    = {},
  TransformOpT transform_op              = {},
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  // Get arch ID
  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

  const rfa_policy active_policy = policy_selector(arch_id);
  const auto tile_items =
    static_cast<OffsetT>(active_policy.single_tile.block_threads * active_policy.single_tile.items_per_thread);

  using deterministic_add_t  = deterministic_sum_t<AccumT>;
  using input_unwrapped_it_t = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;

  input_unwrapped_it_t d_in_unwrapped = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);

  if (num_items <= tile_items)
  {
    return invoke_single_tile<PolicySelector,
                              input_unwrapped_it_t,
                              OutputIteratorT,
                              OffsetT,
                              deterministic_add_t,
                              InitT,
                              typename deterministic_add_t::DeterministicAcc,
                              TransformOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_in_unwrapped,
      d_out,
      num_items,
      deterministic_add_t{},
      init,
      stream,
      transform_op,
      active_policy,
      launcher_factory);
  }

  return invoke_passes<PolicySelector,
                       input_unwrapped_it_t,
                       OutputIteratorT,
                       OffsetT,
                       deterministic_add_t,
                       InitT,
                       typename deterministic_add_t::DeterministicAcc,
                       TransformOpT>(
    d_temp_storage,
    temp_storage_bytes,
    d_in_unwrapped,
    d_out,
    num_items,
    deterministic_add_t{},
    init,
    stream,
    transform_op,
    active_policy,
    launcher_factory);
}
} // namespace detail::rfa
CUB_NAMESPACE_END
