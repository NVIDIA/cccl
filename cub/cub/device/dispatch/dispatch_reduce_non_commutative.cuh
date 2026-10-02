// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce_non_commutative.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/logging.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce_non_commutative.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__functional/identity.h>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_non_commutative
{
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename TransformOpT = ::cuda::std::identity,
          typename AccumT = decltype(reduce::select_accum_t<InputIteratorT, InitValueT, ReductionOpT, TransformOpT>(
            static_cast<use_default*>(nullptr))),
          typename PolicySelector        = reduce::policy_selector_from_types<AccumT, OffsetT, ReductionOpT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires reduce::reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitValueT init,
  cudaStream_t stream,
  TransformOpT transform_op              = {},
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  const ReducePolicy active_policy = policy_selector(cc);
  detail::log_dispatch("DeviceReduceNonCommutative", cc, active_policy);

  using input_it_t          = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<InputIteratorT>;
  const input_it_t d_in_raw = THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(d_in);

  const int single_tile_threads = active_policy.single_tile.threads_per_block;
  const auto single_tile_items =
    static_cast<OffsetT>(single_tile_threads) * static_cast<OffsetT>(active_policy.single_tile.items_per_thread);

  if (num_items <= single_tile_items)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    _CUB_LOG_KERNEL_LAUNCH(
      "device_reduce_non_commutative_single_tile_kernel", 1, 1, 1, single_tile_threads, 0, stream, "");

    if (const auto error = CubDebug(
          launcher_factory(1, single_tile_threads, 0, stream)
            .doit(device_reduce_non_commutative_single_tile_kernel<
                    PolicySelector,
                    input_it_t,
                    OutputIteratorT,
                    OffsetT,
                    ReductionOpT,
                    InitValueT,
                    AccumT,
                    TransformOpT>,
                  d_in_raw,
                  d_out,
                  num_items,
                  reduction_op,
                  init,
                  transform_op)))
    {
      return error;
    }

    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }
    return CubDebug(detail::DebugSyncStream(stream));
  }

  const auto reduce_kernel =
    device_reduce_non_commutative_kernel<PolicySelector, input_it_t, OffsetT, ReductionOpT, AccumT, TransformOpT>;
  const int threads_per_block = active_policy.multi_tile.threads_per_block;

  int sm_count = 0;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
  {
    return error;
  }

  int sm_occupancy = 0;
  if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(sm_occupancy, reduce_kernel, threads_per_block)))
  {
    return error;
  }

  // At least one block, so a kernel that cannot be resident fails at launch instead of dividing by zero here
  const int warps_per_block = threads_per_block / warp_threads;
  const int max_blocks      = ::cuda::std::max(1, sm_occupancy * sm_count * detail::subscription_factor);
  const auto even_share     = even_share_t<OffsetT>::make(
    num_items, max_blocks * warps_per_block, warp_threads * active_policy.multi_tile.items_per_thread);
  const int grid_size = ::cuda::ceil_div(even_share.num_shares, warps_per_block);

  void* allocations[1]             = {};
  const size_t allocation_sizes[1] = {grid_size * sizeof(AccumT)};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  auto* d_block_aggregates = static_cast<AccumT*>(allocations[0]);

  _CUB_LOG_KERNEL_LAUNCH(
    "device_reduce_non_commutative_kernel",
    grid_size,
    1,
    1,
    threads_per_block,
    0,
    stream,
    ", SM occupancy: %d",
    sm_occupancy);

  if (const auto error = CubDebug(
        launcher_factory(grid_size, threads_per_block, 0, stream)
          .doit(reduce_kernel, d_in_raw, d_block_aggregates, even_share, reduction_op, transform_op)))
  {
    return error;
  }

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
  {
    return error;
  }

  _CUB_LOG_KERNEL_LAUNCH(
    "device_reduce_non_commutative_single_tile_kernel", 1, 1, 1, single_tile_threads, 0, stream, "");

  if (const auto error = CubDebug(
        launcher_factory(1, single_tile_threads, 0, stream)
          .doit(device_reduce_non_commutative_single_tile_kernel<
                  PolicySelector,
                  AccumT*,
                  OutputIteratorT,
                  int,
                  ReductionOpT,
                  InitValueT,
                  AccumT,
                  ::cuda::std::identity>,
                d_block_aggregates,
                d_out,
                grid_size,
                reduction_op,
                init,
                ::cuda::std::identity{})))
  {
    return error;
  }

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }
  return CubDebug(detail::DebugSyncStream(stream));
}
} // namespace detail::reduce_non_commutative

CUB_NAMESPACE_END
