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
 * @brief Elementwise transforms over sharded views (in-place unary and
 *        n-ary zip). No cross-place stage: each shard transforms locally.
 *
 * The engine is the MGMN transform (`cuda::experimental::mgmn::transform`,
 * rank-local `cub::DeviceTransform`), instantiated over the in-process
 * `places_communicator` of `mgmn_adapter.cuh` — the lane's owned group for
 * group-built environments — one rank per non-empty shard, the shard
 * environments handed to it as they are. The sharded verbs keep their
 * signatures and their contract; the engine is not visible to the caller.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__iterator/zip_function.h>
#include <cuda/__iterator/zip_iterator.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/transform/transform.h>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/mgmn_adapter.cuh>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cstddef>
#include <vector>

namespace cuda::experimental::sharded
{
namespace reserved
{
//! @brief Is `_Op` an operator whose return type host code cannot query: an
//! extended `__device__` lambda without a trailing return type? (nvcc's
//! host-side stub of such a lambda has no queryable result;
//! `cuda::std::invoke_result` refuses it with a static assertion rather than
//! failing softly.) Named function objects, `__host__ __device__` lambdas
//! and `-> R` lambdas are queryable.
template <class _Op>
inline constexpr bool __opaque_result_v =
#if _CCCL_CUDA_COMPILER(NVCC) && defined(__CUDACC_EXTENDED_LAMBDA__)
  __nv_is_extended_device_lambda_closure_type(_Op) && !__nv_is_extended_host_device_lambda_closure_type(_Op)
  && !__nv_is_extended_device_lambda_with_preserved_return_type(_Op);
#else
  false;
#endif

//! @brief The operator with its result converted to the output element type
//! `_Tp` — the implicit conversion the store `out[i] = op(...)` performs
//! anyway, stated as the return type. That declared type is what lets the
//! engine's host-side checks (`indirectly_unary_invocable`,
//! `indirectly_writable`) accept operators of `__opaque_result_v`. The
//! operator is `mutable`: the host-side stub nvcc gives such a lambda has a
//! non-const call operator.
template <class _Tp, class _Op>
struct __into
{
  mutable _Op __op;

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_HOST_DEVICE_API _Tp operator()(_Args&&... __args) const
  {
    return __op(::cuda::std::forward<_Args>(__args)...);
  }
};

//! @brief @p __op as the engine takes it: itself when its result type is
//! queryable, wrapped in `__into<_Tp>` otherwise.
template <class _Tp, class _Op>
[[nodiscard]] auto __engine_op(_Op __op)
{
  if constexpr (__opaque_result_v<_Op>)
  {
    return __into<_Tp, _Op>{::cuda::std::move(__op)};
  }
  else
  {
    return __op;
  }
}
} // namespace reserved

// ============================================================================
// Concept-generic tier: any sharded_view + per-shard environments
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
 * @throws std::invalid_argument when the environment count does not match
 *         the shard count.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void transform(_S&& data, _Envs&& envs, _UnaryOp op, const _CallEnv& call_env = {})
{
  reserved::__mgmn_map(
    data, envs, call_env, "sharded::transform", [&](const auto& __comms, const auto& __envs, const auto& __lanes) {
      // Output aliases input: the MGMN transform's in-place form.
      const auto __ptrs = reserved::__mgmn_pointers<view_element_t<_S>*>(data, __lanes);
      ::cuda::experimental::mgmn::transform(
        ::cuda::experimental::distributed,
        __comms,
        __envs,
        __ptrs,
        reserved::__mgmn_sizes(data, __lanes),
        __ptrs,
        reserved::__engine_op<view_element_t<_S>>(op));
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

/**
 * @brief N-ary zip transform over sharded views:
 * `out[i] = op(in1[i], in2[i], ...)`, one fused pass per shard (the MGMN
 * transform over a `cuda::zip_iterator` of the shard's input pointers, with
 * the operator lifted to the tuple by `cuda::zip_function`;
 * `cub::DeviceTransform` unwraps that pair into its multi-input kernel).
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
  (reserved::__check_copartitioned(out, ins, "sharded::zip_transform"), ...);

  using __out_t = view_element_t<_SOut>;
  reserved::__mgmn_map(
    out, envs, call_env, "sharded::zip_transform", [&](const auto& __comms, const auto& __envs, const auto& __lanes) {
      // The shard index selects the co-partitioned input shards.
      const auto __inputs = reserved::__mgmn_per_lane(__lanes, [&](::std::size_t __g) {
        return ::cuda::make_zip_iterator(ins.shard(__g).data...);
      });
      ::cuda::experimental::mgmn::transform(
        ::cuda::experimental::distributed,
        __comms,
        __envs,
        __inputs,
        reserved::__mgmn_sizes(out, __lanes),
        reserved::__mgmn_pointers<__out_t*>(out, __lanes),
        ::cuda::make_zip_function(reserved::__engine_op<__out_t>(op)));
    });
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
