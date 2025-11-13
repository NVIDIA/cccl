// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_for.cuh>
#include <cub/device/dispatch/kernels/kernel_for_each.cuh>
#include <cub/device/dispatch/tuning/tuning_for.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{
// The dispatch layer is in the detail namespace until we figure out tuning API
template <class OffsetT, class OpT, class PolicyHubT = policy_hub_t>
struct dispatch_t
{
  OffsetT num_items;
  OpT op;
  cudaStream_t stream;

  CUB_RUNTIME_FUNCTION dispatch_t(OffsetT num_items, OpT op, cudaStream_t stream)
      : num_items(num_items)
      , op(op)
      , stream(stream)
  {}

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_dynamic_block_size()
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int block_threads = 256;
    auto kernel       = detail::for_each::dynamic_kernel<typename PolicyHubT::MaxPolicy, OffsetT, OpT>;
    NV_IF_TARGET(NV_IS_HOST,
                 (int _{}; //
                  if (const auto error = CubDebug(cudaOccupancyMaxPotentialBlockSize(&_, &block_threads, kernel))) {
                    return error;
                  }));

    constexpr int items_per_thread = ActivePolicyT::for_policy_t::items_per_thread;

    const auto tile_size = static_cast<OffsetT>(block_threads * items_per_thread);
    const auto num_tiles = ::cuda::ceil_div(num_items, tile_size);

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking detail::for_each::dynamic_kernel<<<%d, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            static_cast<int>(num_tiles),
            static_cast<int>(block_threads),
            reinterpret_cast<long long>(stream),
            static_cast<int>(items_per_thread));
#endif

    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
            static_cast<unsigned int>(num_tiles), static_cast<unsigned int>(block_threads), 0, stream)
            .doit(kernel, num_items, op)))
    {
      return error;
    }

    if (auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      CubDebug(error = SyncStream(stream)); // TODO(bgruber): this does not make sense to me
      return error;
    }

    return cudaSuccess;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_static_block_size()
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    constexpr int block_threads     = ActivePolicyT::for_policy_t::block_threads;
    constexpr int items_per_thread1 = ActivePolicyT::for_policy_t::items_per_thread;
    constexpr auto tile_size        = static_cast<OffsetT>(block_threads * items_per_thread1);
    const auto num_tiles            = ::cuda::ceil_div(num_items, tile_size);

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking detail::for_each::static_kernel<<<%d, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            static_cast<int>(num_tiles),
            static_cast<int>(block_threads),
            reinterpret_cast<long long>(stream),
            static_cast<int>(items_per_thread1));
#endif

    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
            static_cast<unsigned int>(num_tiles), static_cast<unsigned int>(block_threads), 0, stream)
            .doit(detail::for_each::static_kernel<typename PolicyHubT::MaxPolicy, OffsetT, OpT>, num_items, op)))
    {
      return error;
    }

    if (auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      CubDebug(error = SyncStream(stream)); // TODO(bgruber): this does not make sense to me
      return error;
    }

    return cudaSuccess;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    if constexpr (ActivePolicyT::for_policy_t::block_threads > 0)
    {
      return invoke_static_block_size<ActivePolicyT>();
    }
    else
    {
      return invoke_dynamic_block_size<ActivePolicyT>();
    }
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(OffsetT num_items, OpT op, cudaStream_t stream)
  {
    int ptx_version = 0;
    if (const auto error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    dispatch_t dispatch(num_items, op, stream);
    return CubDebug(PolicyHubT::MaxPolicy::Invoke(ptx_version, dispatch));
  }
};
} // namespace detail::for_each

CUB_NAMESPACE_END
