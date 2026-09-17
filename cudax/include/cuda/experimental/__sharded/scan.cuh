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
 * @brief In-place scans over sharded views.
 *
 * The engine is the MGMN scan (`cuda::experimental::mgmn::inclusive_scan` /
 * `exclusive_scan`: per-rank `cub::DeviceReduce` of the shard, `all_gather`
 * of the P totals, a device prefix over the totals preceding the rank, then
 * the shard's seeded `cub::DeviceScan` in place), instantiated over the
 * in-process `places_communicator` of `mgmn_adapter.cuh`, one rank per shard.
 * Every step is stream work (kernels, copies, event edges): the scans are
 * asynchronous in their stream-bearing form and capture into CUDA graphs;
 * the no-stream form is the synchronous convenience. Algorithm temporaries
 * are drawn from each shard's environment resource. The sharded verbs keep
 * their signatures and their contract; the engine is not visible to the
 * caller.
 *
 * Determinism: the per-shard CUB scans run under `determinism::run_to_run`
 * whenever CUB can honor it for the operator and type (its known operators
 * on integers, `plus` on floating point), or under the requirements the call
 * environment carries (`cuda::execution::require`).
 *
 * Exclusive semantics are the global ones: `out[i] = fold(init,
 * x_0..x_{i-1})` — init enters the fold exactly once.
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

#include <cuda/__functional/operator_properties.h> // identity_element
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/scan/scan.h>
#include <cuda/experimental/__places/place_group.cuh>
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
//! @brief Shared driver of the scans: the distributed MGMN scan of @p __data
//! in place, over every shard, under the sharded contract of `__mgmn_drive`.
//! For the inclusive form @p __init is the identity.
template <bool _Inclusive, class _S, class _Envs, class _ScanOp, class _Tp, class _CallEnv>
_CCCL_HOST_API void __mgmn_scan(
  _S&& __data,
  const _Envs& __envs,
  _ScanOp __op,
  _Tp __init,
  _Tp __identity,
  const _CallEnv& __call_env,
  const char* __what)
{
  const auto __reqs = __mgmn_requirements<__scan_run_to_run_v<_ScanOp, _Tp>>(__call_env);
  __mgmn_drive<true>(
    __data,
    __envs,
    __call_env,
    __what,
    [&](const auto& __env) {
      return __mgmn_alloc_env(__env, __reqs);
    },
    [&](const auto& __comms, const auto& __menvs, const auto& __lanes) {
      const auto __inputs  = __mgmn_pointers<const _Tp*>(__data, __lanes);
      const auto __sizes   = __mgmn_sizes(__data, __lanes);
      const auto __outputs = __mgmn_pointers<_Tp*>(__data, __lanes);
      if constexpr (_Inclusive)
      {
        ::cuda::experimental::mgmn::inclusive_scan(
          ::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, __init, __op, __identity);
      }
      else
      {
        ::cuda::experimental::mgmn::exclusive_scan(
          ::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, __init, __op, __identity);
      }
    });
}
} // namespace reserved

// ============================================================================
// Concept-generic tier: scans over any sharded_view
// ============================================================================

/**
 * @brief In-place inclusive scan over any `sharded_view`:
 * `data[i] = fold(data[0..i])` across the global index space.
 *
 * Contract per the call environment: stream present (`async_call_env`) =
 * asynchronous (lane-ordered by default: enqueue on the environments'
 * streams, cross-shard steps as event edges between them, no host
 * synchronization; `composition::bracketed` on the call environment seals
 * the call against the call stream instead; capture-legal — under capture
 * the lanes must already be capturing, or the call refuses at entry); no
 * stream = synchronous convenience (refused under `sync_policy::forbid` and
 * under capture).
 *
 * @p identity is the operator's identity element, defaulted where
 * `cuda::identity_element` knows the operator; custom operators supply it.
 *
 * @throws std::invalid_argument when the environment count does not match
 *         the shard count, or on more than 64 shards.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__mgmn_scan<true>(
    ::cuda::std::forward<_S>(data), envs, scan_op, identity, identity, call_env, "sharded::inclusive_scan");
}

/// @brief In-place inclusive scan (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__mgmn_scan<true>(
    ::cuda::std::forward<_S>(data), envs, scan_op, identity, identity, call_env, "sharded::inclusive_scan");
}

/**
 * @brief In-place exclusive scan over any `sharded_view`:
 * `data[i] = fold(init, data[0..i-1])` across the global index space — the
 * global semantics, init entering the fold exactly once. Contract as for
 * `inclusive_scan`.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__mgmn_scan<false>(
    ::cuda::std::forward<_S>(data), envs, scan_op, init_value, identity, call_env, "sharded::exclusive_scan");
}

/// @brief In-place exclusive scan (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__mgmn_scan<false>(
    ::cuda::std::forward<_S>(data), envs, scan_op, init_value, identity, call_env, "sharded::exclusive_scan");
}

// Scan conveniences over the generic tier ------------------------------------

/// @brief In-place inclusive sum (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_sum(_S&& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  sharded::inclusive_scan(::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief In-place inclusive sum (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
_CCCL_HOST_API void inclusive_sum(_S&& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  using elem_t    = view_element_t<_S>;
  sharded::inclusive_scan(::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief In-place exclusive sum (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void
exclusive_sum(_S&& data, const _Envs& envs, view_element_t<_S> init_value = {}, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  sharded::exclusive_scan(
    ::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, init_value, elem_t{0}, call_env);
}

/// @brief In-place exclusive sum (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
_CCCL_HOST_API void exclusive_sum(_S&& data, view_element_t<_S> init_value = {}, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  using elem_t    = view_element_t<_S>;
  sharded::exclusive_scan(
    ::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, init_value, elem_t{0}, call_env);
}
} // namespace cuda::experimental::sharded
