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
 * @brief The environment tiers of the sharded concepts: per-shard
 *        environments (`sharded_env`, `sharded_alloc_env`,
 *        `sharded_env_range`, `self_bound`) and the per-call environment
 *        properties (`composition` / `get_composition`, `sync_policy` /
 *        `get_sync_policy`, `async_call_env`, `default_call_env`).
 *
 * See `<cuda/experimental/__sharded/concepts.cuh>` for the design overview
 * of the three concept tiers.
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

#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/experimental/__sharded/concepts/view.cuh>

#include <cstddef>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::sharded
{
template <class _Tp>
_CCCL_CONCEPT __convertible_to_stream_ref = ::cuda::std::convertible_to<_Tp, ::cuda::stream_ref>;

namespace reserved
{
//! @brief Number of environments in a `sharded_env_range`, as `size_t`.
template <class _Envs>
[[nodiscard]] ::std::size_t __env_count(const _Envs& __envs)
{
  return static_cast<::std::size_t>(__envs.size());
}
} // namespace reserved

// ===========================================================================
// Per-shard environments (the binding tier)
// ===========================================================================

//! @brief A per-shard environment: anything the `cuda::get_stream`
//! customization point can extract a stream from (a `.stream()` /
//! `.get_stream()` member, a `query(get_stream_t)` env, or something
//! convertible to `stream_ref`).
template <class _Env>
_CCCL_CONCEPT sharded_env =
  _CCCL_REQUIRES_EXPR((_Env), const _Env& __e)(_Satisfies(__convertible_to_stream_ref)::cuda::get_stream(__e));

template <class _Tp>
_CCCL_CONCEPT __not_void = !::cuda::std::is_void_v<_Tp>;

//! @brief A per-shard environment that can also allocate: additionally
//! answers `cuda::mr::get_memory_resource`. Required by scratch-bearing
//! algorithms (reduce, scan, histogram, ...); the map family needs only
//! `sharded_env`.
template <class _Env>
_CCCL_CONCEPT sharded_alloc_env = _CCCL_REQUIRES_EXPR((_Env), const _Env& __e)(
  requires(sharded_env<_Env>), _Satisfies(__not_void)::cuda::mr::get_memory_resource(__e));

//! @brief An indexed family of per-shard environments: `size()` and
//! `operator[](i)` yielding a `sharded_env`. `envs[i]` binds shard `i`.
template <class _Range>
_CCCL_CONCEPT sharded_env_range = _CCCL_REQUIRES_EXPR((_Range), const _Range& __r)(
  _Satisfies(__convertible_to_size) __r.size(),
  requires(sharded_env<::cuda::std::remove_cvref_t<decltype(__r[::std::size_t{0}])>>));

//! @brief As `sharded_env_range`, with allocating environments.
template <class _Range>
_CCCL_CONCEPT sharded_alloc_env_range = _CCCL_REQUIRES_EXPR((_Range), const _Range& __r)(
  _Satisfies(__convertible_to_size) __r.size(),
  requires(sharded_alloc_env<::cuda::std::remove_cvref_t<decltype(__r[::std::size_t{0}])>>));

//! @brief A self-bound sharded structure: a `sharded_view` for which
//! `default_envs(s)` (found by argument-dependent lookup) yields a
//! `sharded_env_range` with one environment per shard.
//!
//! This is an *optional* capability in the spirit of `std::execution`'s
//! `get_env`: the view concept never stores environments; types built by a
//! provider (containers whose shards recorded their streams and places at
//! construction) can answer the query anyway. Pure transported views do not
//! model it and are used through the explicit-environment overloads.
template <class _S>
_CCCL_CONCEPT self_bound = _CCCL_REQUIRES_EXPR((_S), const _S& __s)(
  requires(sharded_view<_S>), requires(sharded_env_range<::cuda::std::remove_cvref_t<decltype(default_envs(__s))>>));

// ===========================================================================
// Per-call environment (the combine-scope tier)
// ===========================================================================

// ===========================================================================
// The composition property (per-call): lane-ordered (default) or bracketed
// ===========================================================================

//! @brief Per-call composition selector: how an asynchronous call orders
//! against the call environment's stream.
enum class composition
{
  lane_ordered, //!< default: enqueue on the lanes, no call-stream edges
  bracketed //!< fork-all/join-all against the call stream, per call
};

//! @brief Query object for the per-call composition property (defaults to
//! `composition::lane_ordered` when absent).
struct get_composition_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, get_composition_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    return __env.query(*this);
  }
};
_CCCL_GLOBAL_CONSTANT get_composition_t get_composition{};

//! @brief Read the composition property off a call environment
//! (`composition::lane_ordered` when the environment does not carry one).
template <class _CallEnv>
[[nodiscard]] constexpr composition query_composition(const _CallEnv& __env) noexcept
{
  if constexpr (::cuda::std::execution::__queryable_with<_CallEnv, get_composition_t>)
  {
    return __env.query(get_composition);
  }
  else
  {
    (void) __env;
    return composition::lane_ordered;
  }
}

//! @brief Synchronization policy carried by a per-call environment.
enum class sync_policy
{
  allow, //!< best effort: the call may synchronize with the host where the
         //!< algorithm's documented contract says so
  forbid //!< any would-be host synchronization throws `std::runtime_error`
         //!< *before* the blocking call (the capture-guard discipline)
};

//! @brief Query tag for the synchronization policy of a per-call
//! environment: `env.query(get_sync_policy_t{}) -> sync_policy`.
struct get_sync_policy_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, get_sync_policy_t>)
  [[nodiscard]] constexpr sync_policy operator()(const _Env& __env) const noexcept
  {
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_sync_policy_t get_sync_policy{};

//! @brief The synchronization policy of a per-call environment;
//! `sync_policy::allow` when the environment does not carry one.
template <class _CallEnv>
[[nodiscard]] constexpr sync_policy query_sync_policy(const _CallEnv& __env) noexcept
{
  if constexpr (::cuda::std::execution::__queryable_with<_CallEnv, get_sync_policy_t>)
  {
    return __env.query(get_sync_policy_t{});
  }
  else
  {
    (void) __env;
    return sync_policy::allow;
  }
}

//! @brief Does this per-call environment select the asynchronous contract?
//!
//! Presence of a stream (via `cuda::get_stream`) selects it: the call
//! returns after enqueue and performs no host synchronization (for the
//! operations whose documented contract offers the asynchronous form).
//! Ordering follows the composition contract: lane-ordered by default
//! (`composition::lane_ordered`), sealed against the call stream under
//! `composition::bracketed`; combine-bearing terminators deliver their
//! result on the call stream regardless (their edges are definitional).
template <class _CallEnv>
_CCCL_CONCEPT async_call_env = sharded_env<_CallEnv>;

//! @brief An empty per-call environment: synchronous contract, best-effort
//! policy. The default for the convenience overloads.
using default_call_env = ::cuda::std::execution::env<>;
} // namespace cuda::experimental::sharded

// NOLINTEND(bugprone-reserved-identifier)
