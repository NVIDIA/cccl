//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Elementwise transforms over sharded arrays (in-place, unary,
 *        binary). No cross-place stage: each shard transforms locally.
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/stream>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_transform.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <stdexcept>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
// ============================================================================
// Concept-generic tier (pilot): any sharded_view + per-shard environments
// ============================================================================

/**
 * @brief In-place unary transform over any `sharded_view`: for each shard,
 * `data[i] = op(data[i])` on the shard's environment stream.
 *
 * The map family needs no cross-shard stage and no allocation: environments
 * only supply the per-shard stream. Contract, selected by the per-call
 * environment:
 * - `call_env` carries a stream (`async_call_env`): the call enqueues each
 *   shard's work on its environment's stream and touches nothing else
 *   (LANE-ORDERED, the default — consecutive calls on the same environments
 *   are ordered per lane by stream order, independent across lanes),
 *   returns after enqueue, and never synchronizes with the host. A call
 *   environment carrying `composition::bracketed` instead seals the call
 *   against the call stream (fork on entry, join on exit), per call.
 * - `call_env` carries no stream: the call synchronizes the shard
 *   environments' streams before returning (refused when the call
 *   environment carries `sync_policy::forbid`).
 *
 * @throws std::invalid_argument when fewer environments than shards are
 *         supplied.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void transform(_S&& data, _Envs&& envs, _UnaryOp op, const _CallEnv& call_env = {})
{
  __detail::__generic_map(data, envs, call_env, "sharded::transform", [&](const auto& d, cudaStream_t s) {
    thrust::transform(thrust::cuda::par_nosync.on(s), d.data, d.data + d.size, d.data, op);
    cuda_safe_call(cudaGetLastError());
  });
}

/**
 * @brief In-place unary transform over a self-bound sharded structure:
 * environments derived via `default_envs`.
 */
_CCCL_TEMPLATE(class _S, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_UnaryOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_UnaryOp>>))
_CCCL_HOST_API void transform(_S&& data, _UnaryOp op, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::transform(::cuda::std::forward<_S>(data), envs, op, call_env);
}

namespace reserved
{
//! @brief Wrap an n-ary operator so its result rides `cub::DeviceTransform`'s
//! tuple-of-outputs convention (one output here).
template <class _Op>
struct __tuple_result_op
{
  _Op __op;
  template <class... _Args>
  _CCCL_HOST_DEVICE_API auto operator()(_Args&&... __args) const
  {
    return ::cuda::std::tuple<decltype(__op(::cuda::std::forward<_Args>(__args)...))>{
      __op(::cuda::std::forward<_Args>(__args)...)};
  }
};
} // namespace reserved

/**
 * @brief N-ary zip transform over sharded views:
 * `out[i] = op(in1[i], in2[i], ...)`, one fused pass per shard
 * (`cub::DeviceTransform` over a tuple of shard pointers).
 *
 * All views must be co-partitioned with @p out (same shard count, identical
 * per-shard global regions); inputs must be readable where the output's
 * environment executes. An input may be the output (in-place). This is the
 * one-pass form for multi-operand elementwise updates (e.g. a 3-input
 * `w*(c*a + (1-c)*b) + (1-w)*d` style solver step), avoiding the extra
 * memory sweep of chaining binary passes through a temporary.
 *
 * Contract per the call environment, as for `transform`: stream present =
 * asynchronous (lane-ordered by default, `composition::bracketed` to seal
 * the call; no host synchronization); no stream = synchronous convenience
 * (refused under `sync_policy::forbid`).
 *
 * @throws std::invalid_argument on environment shortfall or partition
 *         mismatch.
 */
_CCCL_TEMPLATE(class _SOut, class _Envs, class _Op, class _CallEnv, class... _SIn)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void zip_transform(_SOut&& out, const _Envs& envs, _Op op, const _CallEnv& call_env, const _SIn&... ins)
{
  static_assert(sizeof...(_SIn) >= 1, "zip_transform needs at least one input view");
  const ::std::size_t num_shards = reserved::__shard_count(out);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::zip_transform: fewer environments than shards");
  }
  (reserved::__check_copartitioned(out, ins, "sharded::zip_transform"), ...);

  constexpr bool __is_async = async_call_env<_CallEnv>;

  if constexpr (!__is_async)
  {
    // Refusals first, before any CUDA call: this form synchronizes at the end.
    require_sync_allowed(call_env, "sharded::zip_transform (synchronous form)");
    places::check_not_capturing(nullptr, "sharded::zip_transform");
    for (const auto g : each(num_shards))
    {
      places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::zip_transform");
    }
  }

  for (const auto g : each(num_shards))
  {
    const auto& s_out = out.shard(g);
    if (s_out.size == 0)
    {
      continue;
    }
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    if constexpr (__is_async)
    {
      __detail::__wait_stream_on(shard_stream.get(), ::cuda::get_stream(call_env).get());
    }
    stream_scope scope(shard_stream.get());
    cuda_safe_call(cub::DeviceTransform::Transform(
      ::cuda::std::tuple{ins.shard(g).data...},
      ::cuda::std::tuple{s_out.data},
      s_out.size,
      reserved::__tuple_result_op<_Op>{op},
      envs[g]));
  }
  if constexpr (__is_async)
  {
    for (const auto g : each(num_shards))
    {
      if (out.shard(g).size != 0)
      {
        __detail::__wait_stream_on(::cuda::get_stream(call_env).get(), ::cuda::get_stream(envs[g]).get());
      }
    }
  }

  if constexpr (!__is_async)
  {
    barrier(envs);
  }
}

/**
 * @brief N-ary zip transform over a self-bound output: environments derived
 * via `default_envs(out)`, synchronous convenience.
 */
_CCCL_TEMPLATE(class _SOut, class _Op, class... _SIn)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void zip_transform(_SOut&& out, _Op op, const _SIn&... ins)
{
  const auto envs = default_envs(out);
  sharded::zip_transform(::cuda::std::forward<_SOut>(out), envs, op, default_call_env{}, ins...);
}
} // namespace cuda::experimental::sharded
