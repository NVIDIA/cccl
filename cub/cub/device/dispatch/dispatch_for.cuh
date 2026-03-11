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
#include <cub/detail/arch_dispatch.cuh>
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
template <typename PolicySelector, typename OffsetT, typename OpT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
invoke_dynamic_block_size(OffsetT num_items, OpT op, cudaStream_t stream, for_policy active_policy)
{
  int block_threads = 256;
  auto kernel       = detail::for_each::dynamic_kernel<PolicySelector, OffsetT, OpT>;
  NV_IF_TARGET(NV_IS_HOST,
               (int _{}; //
                if (const auto error = CubDebug(cudaOccupancyMaxPotentialBlockSize(&_, &block_threads, kernel))) {
                  return error;
                }));

  const auto tile_size = static_cast<OffsetT>(block_threads * active_policy.items_per_thread);
  const auto num_tiles = ::cuda::ceil_div(num_items, tile_size);

#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking detail::for_each::dynamic_kernel<<<%d, %d, 0, %lld>>>(), "
          "%d items per thread\n",
          static_cast<int>(num_tiles),
          static_cast<int>(block_threads),
          reinterpret_cast<long long>(stream),
          static_cast<int>(active_policy.items_per_thread));
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

template <class PolicySelector, class OffsetT, class OpT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
invoke_static_block_size(OffsetT num_items, OpT op, cudaStream_t stream, for_policy active_policy)
{
  const int block_threads    = active_policy.block_threads;
  const int items_per_thread = active_policy.items_per_thread;
  const auto tile_size       = static_cast<OffsetT>(block_threads * items_per_thread);
  const auto num_tiles       = ::cuda::ceil_div(num_items, tile_size);

#ifdef CUB_DEBUG_LOG
  _CubLog("Invoking detail::for_each::static_kernel<<<%d, %d, 0, %lld>>>(), "
          "%d items per thread\n",
          static_cast<int>(num_tiles),
          static_cast<int>(block_threads),
          reinterpret_cast<long long>(stream),
          static_cast<int>(items_per_thread));
#endif

  if (const auto error = CubDebug(
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
          static_cast<unsigned int>(num_tiles), static_cast<unsigned int>(block_threads), 0, stream)
          .doit(detail::for_each::static_kernel<PolicySelector, OffsetT, OpT>, num_items, op)))
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

// The dispatch layer is in the detail namespace until we figure out tuning API
template <class OffsetT, class OpT, class PolicySelector = policy_selector>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
dispatch(OffsetT num_items, OpT op, cudaStream_t stream, PolicySelector policy_selector = {})
{
  if (num_items == 0)
  {
    return cudaSuccess;
  }

  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(ptx_arch_id(arch_id)))
  {
    return error;
  }

  return CubDebug(dispatch_arch(policy_selector, arch_id, [&](auto policy_getter) {
    constexpr for_policy active_policy = policy_getter();
    if constexpr (active_policy.block_threads > 0)
    {
      return invoke_static_block_size<PolicySelector>(num_items, op, stream, active_policy);
    }
    else
    {
      return invoke_dynamic_block_size<PolicySelector>(num_items, op, stream, active_policy);
    }
  }));
}
} // namespace detail::for_each

CUB_NAMESPACE_END
