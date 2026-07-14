// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! This file device-wide, parallel operations for computing a reduction across a sequence of data items residing within
//! device-accessible memory. Current reduction operator supported is ``cuda::std::plus``

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
#include <cub/detail/deferred_parameter.cuh>
#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce_deterministic.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/__argument/argument.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::rfa
{
using cuda::execution::determinism::__determinism_t;
using reduce::policy_selector_from_types;

template <typename Invocable, typename InputT>
using transformed_input_t = ::cuda::std::decay_t<::cuda::std::invoke_result_t<Invocable, InputT>>;

template <typename InitValueT, typename InputIteratorT, typename TransformOpT>
using accum_t = ::cuda::std::
  __accumulator_t<::cuda::std::plus<>, InitValueT, transformed_input_t<TransformOpT, it_value_t<InputIteratorT>>>;

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
          typename InitValueT,
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
  InitValueT init,
  cudaStream_t stream,
  TransformOpT transform_op,
  ReducePolicy active_policy,
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
          active_policy.single_tile.threads_per_block,
          (long long) stream,
          active_policy.single_tile.items_per_thread);
#endif // CUB_DEBUG_LOG

  // Invoke single_reduce_sweep_kernel
  if (const auto error = CubDebug(
        launcher_factory(1, active_policy.single_tile.threads_per_block, 0, stream)
          .doit(detail::reduce::DeterministicDeviceReduceSingleTileKernel<
                  PolicySelector,
                  InputIteratorT,
                  OutputIteratorT,
                  ReductionOpT,
                  InitValueT,
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
          typename InitValueT,
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
  InitValueT init,
  cudaStream_t stream,
  TransformOpT transform_op,
  ReducePolicy active_policy,
  KernelLauncherFactory launcher_factory)
{
  // Immediate chunk sizes are passed to the kernels as-is; deferred problem sizes are read on device.
  using num_items_kernel_t = detail::parameter_from_host_t<int, OffsetT>;

  int sm_count;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  KernelConfig reduce_config;
  if (const auto error = CubDebug(reduce_config.__init(
        detail::reduce::DeterministicDeviceReduceKernel<
          PolicySelector,
          InputIteratorT,
          num_items_kernel_t,
          ReductionOpT,
          DeterministicAccumT,
          TransformOpT>,
        active_policy.multi_tile)))
  {
    return error;
  }

  const int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;
  const int max_blocks              = reduce_device_occupancy * detail::subscription_factor;

  const int num_items_per_chunk = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();

  // A deferred problem size cannot be read on the host, so these defaults stand: the kernel consumes the whole
  // problem in a single launch with the worst-case grid, whose surplus blocks exit early.
  int num_chunks           = 1;
  int chunk_grid_size      = max_blocks;
  int partial_chunk_size   = 0;
  bool has_partial_chunk   = false;
  int last_chunk_grid_size = max_blocks;
  if constexpr (!::cuda::args::__traits<OffsetT>::is_deferred)
  {
    num_chunks = static_cast<int>(::cuda::ceil_div(num_items, num_items_per_chunk));

    const int chunk_tile_grid_size = ::cuda::ceil_div(num_items_per_chunk, reduce_config.tile_size);
    chunk_grid_size                = ::cuda::std::min(max_blocks, chunk_tile_grid_size);

    partial_chunk_size                  = num_items % num_items_per_chunk;
    has_partial_chunk                   = partial_chunk_size != 0;
    const int last_chunk_tile_grid_size = ::cuda::ceil_div(partial_chunk_size, reduce_config.tile_size);

    last_chunk_grid_size = ::cuda::std::min(max_blocks, last_chunk_tile_grid_size);
  }

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

    // An immediate problem size is passed as the current chunk size; a deferred problem size is read on device and
    // `num_current_items` holds the worst-case chunk size, which must not be passed to the kernel.
    const auto kernel_num_items = [=] {
      if constexpr (::cuda::args::__traits<OffsetT>::is_deferred)
      {
        return detail::reduce::make_num_items_kernel_arg(num_items);
      }
      else
      {
        return num_current_items;
      }
    }();

// Log device_reduce_sweep_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking DeterministicDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items "
            "per thread, %d SM occupancy\n",
            current_grid_size,
            active_policy.multi_tile.threads_per_block,
            (long long) stream,
            active_policy.multi_tile.items_per_thread,
            reduce_config.sm_occupancy);
#endif // CUB_DEBUG_LOG

    if (const auto error = CubDebug(
          launcher_factory(current_grid_size, active_policy.multi_tile.threads_per_block, 0, stream)
            .doit(detail::reduce::DeterministicDeviceReduceKernel<PolicySelector,
                                                                  InputIteratorT,
                                                                  num_items_kernel_t,
                                                                  ReductionOpT,
                                                                  DeterministicAccumT,
                                                                  TransformOpT>,
                  d_in,
                  d_chunk_block_reductions,
                  kernel_num_items,
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
          active_policy.single_tile.threads_per_block,
          (long long) stream,
          active_policy.single_tile.items_per_thread);
#endif // CUB_DEBUG_LOG

  // Invoke DeterministicDeviceReduceSingleTileKernel/DeterministicDeviceReduceDeferredSingleTileKernel
  const auto second_pass_error = [&] {
    if constexpr (::cuda::args::__traits<OffsetT>::is_deferred)
    {
      return launcher_factory(1, active_policy.single_tile.threads_per_block, 0, stream)
        .doit(detail::reduce::DeterministicDeviceReduceDeferredSingleTileKernel<
                PolicySelector,
                DeterministicAccumT*,
                OutputIteratorT,
                num_items_kernel_t,
                ReductionOpT,
                InitValueT,
                DeterministicAccumT>,
              d_block_reductions,
              d_out,
              detail::reduce::make_num_items_kernel_arg(num_items),
              reduce_grid_size,
              reduction_op,
              init,
              ::cuda::std::identity{});
    }
    else
    {
      return launcher_factory(1, active_policy.single_tile.threads_per_block, 0, stream)
        .doit(detail::reduce::DeterministicDeviceReduceSingleTileKernel<
                PolicySelector,
                DeterministicAccumT*,
                OutputIteratorT,
                ReductionOpT,
                InitValueT,
                DeterministicAccumT>,
              d_block_reductions,
              d_out,
              reduce_grid_size,
              reduction_op,
              init,
              ::cuda::std::identity{});
    }
  }();
  if (const auto error = CubDebug(second_pass_error))
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
          typename InitValueT,
          typename TransformOpT          = ::cuda::std::identity,
          typename AccumT                = accum_t<InitValueT, InputIteratorT, TransformOpT>,
          typename PolicySelector        = policy_selector_from_types<AccumT,
                                                                      reduce::num_items_offset_t<OffsetT>,
                                                                      deterministic_sum_t<AccumT>,
                                                                      __determinism_t::__gpu_to_gpu>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  InitValueT init                        = {},
  cudaStream_t stream                    = {},
  TransformOpT transform_op              = {},
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  // Get CC
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  const ReducePolicy active_policy = policy_selector(cc);

#if _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::stringstream ss;
                 ss << active_policy;
                 _CubLog("Dispatching DeviceReduceDeterministic to compute capability %d.%d with tuning: %s\n",
                         cc.major_cap(),
                         cc.minor_cap(),
                         ss.str().c_str());
               }))
#endif // _CCCL_HOSTED() && defined(CUB_DEBUG_LOG)

  using deterministic_add_t  = deterministic_sum_t<AccumT>;
  using input_unwrapped_it_t = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;

  input_unwrapped_it_t d_in_unwrapped = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);

  // A deferred problem size cannot be compared against the single-tile capacity on the host, so a deferred
  // reduction always takes the two-pass path.
  if constexpr (!::cuda::args::__traits<OffsetT>::is_deferred)
  {
    const auto tile_items = static_cast<OffsetT>(active_policy.single_tile.threads_per_block)
                          * static_cast<OffsetT>(active_policy.single_tile.items_per_thread);

    if (num_items <= tile_items)
    {
      return invoke_single_tile<PolicySelector,
                                input_unwrapped_it_t,
                                OutputIteratorT,
                                OffsetT,
                                deterministic_add_t,
                                InitValueT,
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
  }

  return invoke_passes<PolicySelector,
                       input_unwrapped_it_t,
                       OutputIteratorT,
                       OffsetT,
                       deterministic_add_t,
                       InitValueT,
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
