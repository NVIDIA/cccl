/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <cub/device/dispatch/tuning/tuning_for.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace for_each
{

template <class Fn>
struct first_parameter
{
  using type = void;
};

template <class C, class R, class A>
struct first_parameter<R (C::*)(A)>
{
  using type = A;
};

template <class C, class R, class A>
struct first_parameter<R (C::*)(A) const>
{
  using type = A;
};

template <class Fn>
using first_parameter_t = typename first_parameter<decltype(&Fn::operator())>::type;

template <class Value, class Fn, class = void>
struct has_unique_value_overload : ::cuda::std::false_type
{};

// clang-format off
template <class Value, class Fn>
struct has_unique_value_overload<
  Value,
  Fn,
  typename ::cuda::std::enable_if<
              !::cuda::std::is_reference<first_parameter_t<Fn>>::value && 
              ::cuda::std::is_convertible<Value, first_parameter_t<Fn>
             >::value>::type>
    : ::cuda::std::true_type
{};

// For trivial types, foreach is not allowed to copy values, even if those are trivially copyable.
// This can be observable if the unary operator takes parameter by reference and modifies it or uses address. 
// The trait below checks if the freedom to copy trivial types can be regained. 
template <typename Value, typename Fn>
using can_regain_copy_freedom = 
  ::cuda::std::integral_constant<
    bool,
    ::cuda::std::is_trivially_constructible<Value>::value && 
    ::cuda::std::is_trivially_copy_assignable<Value>::value && 
    :: cuda::std::is_trivially_move_assignable<Value>::value && 
    ::cuda::std::is_trivially_destructible<Value>::value && 
    has_unique_value_overload<Value, Fn>::value>;
// clang-format on

// This kernel is used when the block size is not known at compile time
template <class ChainedPolicyT, class OffsetT, class OpT>
CUB_DETAIL_KERNEL_ATTRIBUTES void dynamic_kernel(OffsetT num_items, OpT op)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using agent_t         = agent_block_striped_t<active_policy_t, OffsetT, OpT>;

  const auto block_threads  = static_cast<OffsetT>(blockDim.x);
  const auto items_per_tile = active_policy_t::items_per_thread * block_threads;
  const auto tile_base      = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining  = num_items - tile_base;
  const auto items_in_tile  = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {
    agent_t{tile_base, op}.template consume_tile<true>(items_per_tile, block_threads);
  }
  else
  {
    agent_t{tile_base, op}.template consume_tile<false>(items_in_tile, block_threads);
  }
}

// This kernel is used when the block size is known at compile time
template <class ChainedPolicyT, class OffsetT, class OpT>
CUB_DETAIL_KERNEL_ATTRIBUTES //
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads) //
  void static_kernel(OffsetT num_items, OpT op)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using agent_t         = agent_block_striped_t<active_policy_t, OffsetT, OpT>;

  constexpr auto block_threads  = active_policy_t::block_threads;
  constexpr auto items_per_tile = active_policy_t::items_per_thread * block_threads;

  const auto tile_base      = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining  = num_items - tile_base;
  const auto items_in_tile  = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {
    agent_t{tile_base, op}.template consume_tile<true>(items_per_tile, block_threads);
  }
  else
  {
    agent_t{tile_base, op}.template consume_tile<false>(items_in_tile, block_threads);
  }
}

// The dispatch layer is in the detail namespace until we figure out tuning API
template <class OffsetT, class OpT, class PolicyHubT = policy_hub_t>
struct dispatch_t : PolicyHubT
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
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
    cudaError_t Invoke(::cuda::std::false_type /* block size is not known at compile time */)
  {
    using max_policy_t = typename dispatch_t::MaxPolicy;

    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int block_threads = 256;
    cudaError_t error = cudaSuccess;

    NV_IF_TARGET(NV_IS_HOST,
                 (int _{}; //
                  error = cudaOccupancyMaxPotentialBlockSize(
                    &_, &block_threads, detail::for_each::dynamic_kernel<max_policy_t, OffsetT, OpT>);));

    error = CubDebug(error);
    if (cudaSuccess != error)
    {
      return error;
    }

    constexpr int items_per_thread = ActivePolicyT::for_policy_t::items_per_thread;

    const auto tile_size = static_cast<OffsetT>(block_threads * items_per_thread);
    const auto num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking detail::for_each::dynamic_kernel<<<%d, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            static_cast<int>(num_tiles),
            static_cast<int>(block_threads),
            reinterpret_cast<long long>(stream),
            static_cast<int>(items_per_thread));
#endif

    error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
              static_cast<unsigned int>(num_tiles), static_cast<unsigned int>(block_threads), 0, stream)
              .doit(detail::for_each::dynamic_kernel<max_policy_t, OffsetT, OpT>, num_items, op);
    error = CubDebug(error);
    if (cudaSuccess != error)
    {
      return error;
    }

    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      CubDebug(error = SyncStream(stream));
    }

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE
    cudaError_t Invoke(::cuda::std::true_type /* block size is known at compile time */)
  {
    using max_policy_t = typename dispatch_t::MaxPolicy;

    if (num_items == 0)
    {
      return cudaSuccess;
    }

    cudaError_t error              = cudaSuccess;
    constexpr int block_threads    = ActivePolicyT::for_policy_t::block_threads;
    constexpr int items_per_thread = ActivePolicyT::for_policy_t::items_per_thread;

    const auto tile_size = static_cast<OffsetT>(block_threads * items_per_thread);
    const auto num_tiles = cub::DivideAndRoundUp(num_items, tile_size);

#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
    _CubLog("Invoking detail::for_each::static_kernel<<<%d, %d, 0, %lld>>>(), "
            "%d items per thread\n",
            static_cast<int>(num_tiles),
            static_cast<int>(block_threads),
            reinterpret_cast<long long>(stream),
            static_cast<int>(items_per_thread));
#endif

    error = THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
              static_cast<unsigned int>(num_tiles), static_cast<unsigned int>(block_threads), 0, stream)
              .doit(detail::for_each::static_kernel<max_policy_t, OffsetT, OpT>, num_items, op);
    error = CubDebug(error);
    if (cudaSuccess != error)
    {
      return error;
    }

    error = CubDebug(detail::DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      CubDebug(error = SyncStream(stream));
    }

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    constexpr bool static_block_size = ActivePolicyT::for_policy_t::block_threads > 0;
    return Invoke<ActivePolicyT>(::cuda::std::integral_constant<bool, static_block_size>{});
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(OffsetT num_items, OpT op, cudaStream_t stream)
  {
    using max_policy_t = typename dispatch_t::MaxPolicy;

    int ptx_version   = 0;
    cudaError_t error = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    dispatch_t dispatch(num_items, op, stream);

    error = CubDebug(max_policy_t::Invoke(ptx_version, dispatch));

    return error;
  }
};

} // namespace for_each

} // namespace detail

CUB_NAMESPACE_END
