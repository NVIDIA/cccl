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
 * @brief Elementwise generators over sharded arrays: fill, sequence, iota,
 *        tabulate, generate, for_each.
 *
 * These algorithms have no cross-place stage: each shard runs the device
 * primitive locally on its own place and stream; global indices are recovered
 * from the shard's `global_offset`.
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

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
template <typename _Tp>
struct sequence_fn
{
  _Tp start;
  _Tp step;
  size_t global_offset;

  _CCCL_HOST_DEVICE _Tp operator()(size_t idx) const
  {
    return start + static_cast<_Tp>((global_offset + idx) * step);
  }
};

template <typename _Fn>
struct tabulate_fn
{
  _Fn f;
  size_t global_offset;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE auto operator()(size_t idx) const
  {
    return f(global_offset + idx);
  }
};

template <typename _Tp, typename _Op>
struct for_each_fn
{
  _Op op;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void operator()(thrust::tuple<_Tp&, size_t> t) const
  {
    op(thrust::get<0>(t), thrust::get<1>(t));
  }
};
} // namespace reserved

// ============================================================================
// Concept-generic tier: the elementwise family over any sharded_view
// ============================================================================
//
// The per-call environment selects the contract (stream present =
// asynchronous, lane-ordered by default per the composition contract —
// `composition::bracketed` seals a call against the call stream; no stream =
// synchronous convenience; sync_policy::forbid refuses the synchronous
// form). Explicit-environment overloads serve any
// sharded_view; the self-bound overloads derive environments via
// default_envs.

//! @brief Set every element to @p value (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void fill(_S&& data, const _Envs& envs, const _Tp& value, const _CallEnv& call_env = {})
{
  __detail::__generic_map(data, envs, call_env, "sharded::fill", [&](const auto& d, cudaStream_t s) {
    thrust::fill(thrust::cuda::par_nosync.on(s), d.data, d.data + d.size, value);
    cuda_safe_call(cudaGetLastError());
  });
}

//! @brief Set every element to @p value (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_Tp>>))
_CCCL_HOST_API void fill(_S&& data, const _Tp& value, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::fill(::cuda::std::forward<_S>(data), envs, value, call_env);
}

//! @brief data[i] = start + i * step for the GLOBAL index i (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void sequence(
  _S&& data,
  const _Envs& envs,
  view_element_t<_S> start = {},
  view_element_t<_S> step  = view_element_t<_S>{1},
  const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  __detail::__generic_map(data, envs, call_env, "sharded::sequence", [&](const auto& d, cudaStream_t s) {
    thrust::tabulate(thrust::cuda::par_nosync.on(s),
                     d.data,
                     d.data + d.size,
                     reserved::sequence_fn<elem_t>{start, step, static_cast<size_t>(d.global_offset)});
    cuda_safe_call(cudaGetLastError());
  });
}

//! @brief data[i] = start + i for the GLOBAL index i (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
_CCCL_HOST_API void iota(_S&& data, view_element_t<_S> start = {}, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::sequence(::cuda::std::forward<_S>(data), envs, start, view_element_t<_S>{1}, call_env);
}

//! @brief data[i] = f(i) for the GLOBAL index i; `f` device-callable (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _Fn, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void tabulate(_S&& data, const _Envs& envs, _Fn f, const _CallEnv& call_env = {})
{
  __detail::__generic_map(data, envs, call_env, "sharded::tabulate", [&](const auto& d, cudaStream_t s) {
    thrust::tabulate(thrust::cuda::par_nosync.on(s),
                     d.data,
                     d.data + d.size,
                     reserved::tabulate_fn<_Fn>{f, static_cast<size_t>(d.global_offset)});
    cuda_safe_call(cudaGetLastError());
  });
}

//! @brief data[i] = f(i) (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Fn, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_Fn>>))
_CCCL_HOST_API void tabulate(_S&& data, _Fn f, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::tabulate(::cuda::std::forward<_S>(data), envs, f, call_env);
}

//! @brief data[i] = gen() with a stateless, device-callable generator (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _Gen, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void generate(_S&& data, const _Envs& envs, _Gen gen, const _CallEnv& call_env = {})
{
  __detail::__generic_map(data, envs, call_env, "sharded::generate", [&](const auto& d, cudaStream_t s) {
    thrust::generate(thrust::cuda::par_nosync.on(s), d.data, d.data + d.size, gen);
    cuda_safe_call(cudaGetLastError());
  });
}

//! @brief data[i] = gen() (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Gen, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_Gen>>))
_CCCL_HOST_API void generate(_S&& data, _Gen gen, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::generate(::cuda::std::forward<_S>(data), envs, gen, call_env);
}

//! @brief Apply `op(element&, global_index)` to every element (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _Op, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void for_each(_S&& data, const _Envs& envs, _Op op, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  __detail::__generic_map(data, envs, call_env, "sharded::for_each", [&](const auto& d, cudaStream_t s) {
    const size_t global_offset = static_cast<size_t>(d.global_offset);
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d.data, thrust::make_counting_iterator(global_offset)));
    auto end   = thrust::make_zip_iterator(
      thrust::make_tuple(d.data + d.size, thrust::make_counting_iterator(global_offset + d.size)));
    thrust::for_each(thrust::cuda::par_nosync.on(s), begin, end, reserved::for_each_fn<elem_t, _Op>{op});
    cuda_safe_call(cudaGetLastError());
  });
}

//! @brief Apply `op(element&, global_index)` (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Op, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_Op>>))
_CCCL_HOST_API void for_each(_S&& data, _Op op, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::for_each(::cuda::std::forward<_S>(data), envs, op, call_env);
}
} // namespace cuda::experimental::sharded
