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
 * @brief Reference implementation of the map family (`transform`,
 *        `zip_transform`) on the MGMN engine, kept as the comparison point
 *        for interfacing cost (compile time, verbosity, graph shape) against
 *        the direct bodies of `transform.cuh`; not public API.
 *
 * The live sharded transforms are direct per-shard launches: the map family
 * never communicates, so routing it through a communicator-based engine
 * buys nothing at run time (same kernels, same graph) and costs compile
 * time and a longer body. The verbs here keep the exact signatures and
 * contract of the live ones and run `cuda::experimental::mgmn::transform`
 * (rank-local `cub::DeviceTransform`) over the in-process
 * `places_communicator` of `engine/mgmn.cuh` — the lane's owned group for
 * group-built environments — one rank per non-empty shard, the shard
 * environments handed to the engine as they are. That is the simplest
 * algorithm the engines can host, which is what isolates the pure cost of
 * the interface. Measured (and to be re-measured whenever the adapter or
 * the engines change) by compiling `test/sharded/algorithms/elementwise.cu`
 * against each spelling.
 *
 * Namespace `cuda::experimental::sharded::reserved::mgmn_engine`. Not
 * included by `cuda/experimental/sharded.cuh`: include this header
 * explicitly. Parity with the live verbs is checked bitwise by
 * `test/sharded/algorithms/engine_parity.cu`.
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
#include <cuda/experimental/__sharded/composition/verbs.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/container/default_envs.cuh>
#include <cuda/experimental/__sharded/engine/mgmn.cuh>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cstddef>
#include <vector>

namespace cuda::experimental::sharded::reserved::mgmn_engine
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

/**
 * @brief In-place unary transform over any `sharded_view` on the MGMN
 * engine: same signature and contract as `sharded::transform` (see
 * `transform.cuh`), body = `mgmn::transform` in its in-place form (output
 * aliases input), one rank per non-empty shard.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void transform(_S&& data, _Envs&& envs, _UnaryOp op, const _CallEnv& call_env = {})
{
  reserved::__mgmn_map(
    data, envs, call_env, "sharded::transform", [&](const auto& __comms, const auto& __envs, const auto& __lanes) {
      const auto __ptrs = reserved::__mgmn_pointers<view_element_t<_S>*>(data, __lanes);
      ::cuda::experimental::mgmn::transform(
        ::cuda::experimental::distributed,
        __comms,
        __envs,
        __ptrs,
        reserved::__mgmn_sizes(data, __lanes),
        __ptrs,
        mgmn_engine::__engine_op<view_element_t<_S>>(op));
    });
}

/**
 * @brief In-place unary transform over a self-bound sharded structure on
 * the MGMN engine: environments derived via `default_envs`.
 */
_CCCL_TEMPLATE(class _S, class _UnaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_UnaryOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_UnaryOp>>))
_CCCL_HOST_API void transform(_S&& data, _UnaryOp op, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  mgmn_engine::transform(::cuda::std::forward<_S>(data), envs, op, call_env);
}

/**
 * @brief N-ary zip transform over sharded views on the MGMN engine: same
 * signature and contract as `sharded::zip_transform` (see `transform.cuh`),
 * body = `mgmn::transform` over a `cuda::zip_iterator` of the shard's input
 * pointers, with the operator lifted to the tuple by `cuda::zip_function`
 * (`cub::DeviceTransform` unwraps that pair into its multi-input kernel).
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
        ::cuda::make_zip_function(mgmn_engine::__engine_op<__out_t>(op)));
    });
}

/**
 * @brief N-ary zip transform over a self-bound output on the MGMN engine:
 * environments derived via `default_envs(out)`, synchronous convenience.
 */
_CCCL_TEMPLATE(class _SOut, class _Op, class... _SIn)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void zip_transform(_SOut&& out, _Op op, const _SIn&... ins)
{
  const auto envs = default_envs(out);
  mgmn_engine::zip_transform(::cuda::std::forward<_SOut>(out), envs, op, default_call_env{}, ins...);
}
} // namespace cuda::experimental::sharded::reserved::mgmn_engine
