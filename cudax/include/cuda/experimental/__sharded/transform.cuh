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
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <stdexcept>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/// @brief In-place unary transform: data[i] = op(data[i]).
template <typename _Tp, typename _UnaryOp>
_CCCL_HOST_API void transform(place_group&, sharded_array<_Tp>& data, _UnaryOp op, bool blocking = true)
{
  if (data.empty())
  {
    return;
  }

  data.each_shard->*[op](auto& s) {
    thrust::transform(thrust::cuda::par_nosync.on(s.stream), s.data, s.data + s.size, s.data, op);
    cuda_safe_call(cudaGetLastError());
  };

  if (blocking)
  {
    data.sync();
  }
}

/**
 * @brief Out-of-place unary transform: output[i] = op(input[i]).
 *
 * Input and output must be compatible (same shard sizes and places); each
 * output stream waits on the corresponding input stream.
 *
 * @throws std::invalid_argument when the layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _UnaryOp>
_CCCL_HOST_API void
transform(place_group&, const sharded_array<_Tp>& input, sharded_array<_Up>& output, _UnaryOp op, bool blocking = true)
{
  check_compatible(input, output, "transform (unary out-of-place)");

  if (input.empty())
  {
    return;
  }

  // Make each output stream wait for the corresponding input stream
  for (size_t g = 0; g < input.num_shards(); g++)
  {
    ::cuda::stream_ref{output.shard(g).stream}.wait(::cuda::stream_ref{input.shard(g).stream});
  }

  output.each_shard->*[&input, op](const size_t g, auto& out_shard) {
    const auto& in_shard = input.shard(g);
    thrust::transform(
      thrust::cuda::par_nosync.on(out_shard.stream), in_shard.data, in_shard.data + in_shard.size, out_shard.data, op);
    cuda_safe_call(cudaGetLastError());
  };

  if (blocking)
  {
    output.sync();
  }
}

/**
 * @brief Binary transform: output[i] = op(input1[i], input2[i]).
 *
 * All three arrays must be compatible (same shard sizes and places).
 *
 * @throws std::invalid_argument when the layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _BinaryOp>
_CCCL_HOST_API void transform(
  place_group&,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Tp>& input2,
  sharded_array<_Up>& output,
  _BinaryOp op,
  bool blocking = true)
{
  check_compatible(input1, input2, "transform (binary): input1 vs input2");
  check_compatible(input1, output, "transform (binary): inputs vs output");

  if (input1.empty())
  {
    return;
  }

  // Make each output stream wait for both input streams
  for (size_t g = 0; g < input1.num_shards(); g++)
  {
    const auto& out_shard = output.shard(g);
    ::cuda::stream_ref{out_shard.stream}.wait(::cuda::stream_ref{input1.shard(g).stream});
    ::cuda::stream_ref{out_shard.stream}.wait(::cuda::stream_ref{input2.shard(g).stream});
  }

  output.each_shard->*[&input1, &input2, op](const size_t g, auto& out_shard) {
    const auto& in1_shard = input1.shard(g);
    const auto& in2_shard = input2.shard(g);
    thrust::transform(
      thrust::cuda::par_nosync.on(out_shard.stream),
      in1_shard.data,
      in1_shard.data + in1_shard.size,
      in2_shard.data,
      out_shard.data,
      op);
    cuda_safe_call(cudaGetLastError());
  };

  if (blocking)
  {
    output.sync();
  }
}

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
 * - `call_env` carries a stream (`async_call_env`): the call is ordered
 *   against that stream (fork on entry, join on exit), returns after
 *   enqueue, and never synchronizes with the host.
 * - `call_env` carries no stream: the call synchronizes the shard
 *   environments' streams before returning (refused when the call
 *   environment carries `sync_policy::forbid`).
 *
 * @throws std::invalid_argument when fewer environments than shards are
 *         supplied.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
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
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_env_range<::cuda::std::remove_cvref_t<_UnaryOp>>) _CCCL_AND(
  !sharded_view<::cuda::std::remove_cvref_t<_UnaryOp>>))
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

//! @brief Check that two sharded views are co-partitioned: same shard count
//! and, per shard, identical global regions.
template <class _SA, class _SB>
void __check_copartitioned(const _SA& __a, const _SB& __b, const char* __what)
{
  if (static_cast<::std::size_t>(__a.num_shards()) != static_cast<::std::size_t>(__b.num_shards()))
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": shard count mismatch");
  }
  for (::std::size_t __g = 0; __g < static_cast<::std::size_t>(__a.num_shards()); ++__g)
  {
    if (static_cast<::std::size_t>(__a.shard(__g).size) != static_cast<::std::size_t>(__b.shard(__g).size)
        || static_cast<::std::size_t>(__a.shard(__g).global_offset)
             != static_cast<::std::size_t>(__b.shard(__g).global_offset))
    {
      _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": shard regions differ (not co-partitioned)");
    }
  }
}
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
 * asynchronous (fork/join against it, no host synchronization); no stream =
 * synchronous convenience (refused under `sync_policy::forbid`).
 *
 * @throws std::invalid_argument on environment shortfall or partition
 *         mismatch.
 */
_CCCL_TEMPLATE(class _SOut, class _Envs, class _Op, class _CallEnv, class... _SIn)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
                 sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void
zip_transform(_SOut&& out, const _Envs& envs, _Op op, const _CallEnv& call_env, const _SIn&... ins)
{
  static_assert(sizeof...(_SIn) >= 1, "zip_transform needs at least one input view");
  const ::std::size_t num_shards = static_cast<::std::size_t>(out.num_shards());
  if (static_cast<::std::size_t>(envs.size()) < num_shards)
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
    for (::std::size_t g = 0; g < num_shards; g++)
    {
      places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::zip_transform");
    }
  }

  for (::std::size_t g = 0; g < num_shards; g++)
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
    for (::std::size_t g = 0; g < num_shards; g++)
    {
      if (out.shard(g).size != 0)
      {
        __detail::__wait_stream_on(::cuda::get_stream(call_env).get(), ::cuda::get_stream(envs[g]).get());
      }
    }
  }

  if constexpr (!__is_async)
  {
    for (::std::size_t g = 0; g < num_shards; g++)
    {
      if (out.shard(g).size != 0)
      {
        cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[g]).get()));
      }
    }
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
