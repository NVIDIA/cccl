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
 * @brief The composition vocabulary: how sharded asynchronous calls order
 *        against each other, and the explicit synchronization verbs.
 *
 * A LANE is one ordering domain — one stream per place, the environments'
 * streams. The composition contract has one rule: asynchronous calls on the
 * same environments are ordered PER LANE by stream order, and independent
 * across lanes. A call enqueues each shard's work on `envs[i]` and touches
 * nothing else; everything beyond stream order is said explicitly, with the
 * verbs below. (Results attach to their OUTPUT's timeline: after an
 * asynchronous `reduce_into`, the aggregate is ready in stream order on the
 * call environment's stream — awaiting a result means awaiting the stream
 * it was delivered on, not the lanes that produced it.)
 *
 * The verbs are concepts-tier free functions: their only requirement is an
 * environment range answering `cuda::get_stream`, so any binding — the
 * containers' `default_envs`, `place_group::envs(lane_id)`, foreign
 * user-built ranges — gets them unchanged. Providers manufacture
 * environments; composition needs only their streams.
 *
 * Opting back into sealed calls: a call environment carrying
 * `composition::bracketed` (the `get_composition` query) restores the
 * fork-all/join-all bracket around one call — every shard's work then waits
 * for the call stream and the call stream waits for every shard: the
 * foreign-stream composition case, paid per call.
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

#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <initializer_list>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
// ===========================================================================
// The synchronization verbs (concepts-tier free functions over env ranges)
// ===========================================================================

//! @brief Host barrier: synchronize every lane with the host.
//!
//! The explicit global join. Refuses (before any CUDA call) under
//! `sync_policy::forbid` on @p policy_env and under CUDA graph capture.
_CCCL_TEMPLATE(class _Envs, class _PolicyEnv = default_call_env)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>> _CCCL_AND(
  !::cuda::std::convertible_to<_PolicyEnv, ::cuda::stream_ref>))
_CCCL_HOST_API void barrier(const _Envs& envs, const _PolicyEnv& policy_env = {})
{
  require_sync_allowed(policy_env, "sharded::barrier");
  const ::std::size_t n = reserved::__env_count(envs);
  places::check_not_capturing(nullptr, "sharded::barrier");
  for (const auto i : each(n))
  {
    places::check_not_capturing(::cuda::get_stream(envs[i]).get(), "sharded::barrier");
  }
  for (const auto i : each(n))
  {
    cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[i]).get()));
  }
}

//! @brief Stream barrier: make @p stream wait for all work currently
//! enqueued on every lane (event edges; non-blocking, capture-legal).
//!
//! The pipeline-boundary form: join the lanes into one timeline — a
//! caller's stream, a capture origin, a communicator's stream — without
//! touching the host.
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void barrier(const _Envs& envs, ::cuda::stream_ref stream)
{
  for (const auto i : each(reserved::__env_count(envs)))
  {
    __detail::__wait_stream_on(stream.get(), ::cuda::get_stream(envs[i]).get());
  }
}

//! @brief Declare a cross-lane dependency: lane @p target waits for all
//! work currently enqueued on each lane in @p sources (event edges;
//! non-blocking, capture-legal by construction).
//!
//! The per-lane, caller-chosen spelling of what the bracket did globally.
//! A forgotten `lane_wait` between genuinely coupled lanes is a race — the
//! same honesty as any stream programming.
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void lane_wait(const _Envs& envs, ::std::size_t target, ::std::initializer_list<::std::size_t> sources)
{
  const ::std::size_t n = reserved::__env_count(envs);
  if (target >= n)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::lane_wait: target lane out of range");
  }
  const cudaStream_t target_stream = ::cuda::get_stream(envs[target]).get();
  for (const ::std::size_t s : sources)
  {
    if (s >= n)
    {
      _CCCL_THROW(::std::invalid_argument, "sharded::lane_wait: source lane out of range");
    }
    __detail::__wait_stream_on(target_stream, ::cuda::get_stream(envs[s]).get());
  }
}

//! @brief As above, with a second environment range for the source lanes
//! (cross-field coupling: `lane_wait(envs_y, i, envs_x, {i})` makes field
//! y's lane `i` wait for field x's lane `i`).
_CCCL_TEMPLATE(class _EnvsTo, class _EnvsFrom)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_EnvsTo>> _CCCL_AND
                 sharded_env_range<::cuda::std::remove_cvref_t<_EnvsFrom>>)
_CCCL_HOST_API void lane_wait(
  const _EnvsTo& envs_to,
  ::std::size_t target,
  const _EnvsFrom& envs_from,
  ::std::initializer_list<::std::size_t> sources)
{
  if (target >= reserved::__env_count(envs_to))
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::lane_wait: target lane out of range");
  }
  const cudaStream_t target_stream = ::cuda::get_stream(envs_to[target]).get();
  const ::std::size_t nf           = reserved::__env_count(envs_from);
  for (const ::std::size_t s : sources)
  {
    if (s >= nf)
    {
      _CCCL_THROW(::std::invalid_argument, "sharded::lane_wait: source lane out of range");
    }
    __detail::__wait_stream_on(target_stream, ::cuda::get_stream(envs_from[s]).get());
  }
}

//! @brief Host-synchronize ONE lane. Refuses under `sync_policy::forbid`
//! on @p policy_env and under capture, before any CUDA call.
_CCCL_TEMPLATE(class _Envs, class _PolicyEnv = default_call_env)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void lane_sync(const _Envs& envs, ::std::size_t i, const _PolicyEnv& policy_env = {})
{
  require_sync_allowed(policy_env, "sharded::lane_sync");
  if (i >= reserved::__env_count(envs))
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::lane_sync: lane out of range");
  }
  const cudaStream_t s = ::cuda::get_stream(envs[i]).get();
  places::check_not_capturing(nullptr, "sharded::lane_sync");
  places::check_not_capturing(s, "sharded::lane_sync");
  cuda_safe_call(cudaStreamSynchronize(s));
}
} // namespace cuda::experimental::sharded
