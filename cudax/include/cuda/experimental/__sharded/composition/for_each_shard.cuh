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
 * @brief `for_each_shard`: run a caller-supplied per-shard body under the
 *        sharded call contract — the map family's driver as a public verb.
 *
 * Every algorithm of the map family (`fill`, `transform`, `for_each`,
 * `segmented_reduce`, ...) is a per-shard launch handed to one shared driver
 * that applies the call contract around it: visit every non-empty shard, on
 * its environment's stream, under `stream_scope` (so a launch lands in the
 * shard's execution context, e.g. a green-context place), with the
 * environment-count guard, lane-ordered or `composition::bracketed`
 * composition against the call stream, the synchronous no-stream form
 * (refused under `sync_policy::forbid`), and the capture-time refusal.
 *
 * `for_each_shard` exposes that driver. It is the extension point for work
 * the algorithm tier does not name: a CUB or cuco call per shard, a
 * hand-written or generated kernel per shard, a per-shard reduction whose P
 * results the caller wants as a vector (the first half of `reduce`), a
 * per-shard scan with no cross-shard carry, or the emit step of a
 * data-dependent expansion. The body only *enqueues*; the driver owns the
 * ordering.
 *
 * Body arities (host callable, chosen by invocability, most informative
 * first):
 *
 *   - `(size_t g, const descriptor& d, const Env& env)` — the shard index,
 *     the shard descriptor (`data`, `size`, `global_offset`, `place`) and
 *     the shard's environment (`cuda::get_stream(env)`; and, for
 *     `sharded_alloc_env`, `cuda::mr::get_memory_resource(env)` — draw
 *     stream-ordered scratch from it so temporaries land on the shard's
 *     place);
 *   - `(size_t g, const descriptor& d, cudaStream_t s)`;
 *   - `(const descriptor& d, cudaStream_t s)`.
 *
 * Contract, identical to the rest of the map family:
 *
 *   - empty shards are skipped (the body never sees `size == 0`);
 *   - with a stream-bearing call environment the call is asynchronous and
 *     LANE-ORDERED: work is enqueued on each shard's stream and nothing else
 *     is touched; `composition::bracketed` on the call environment adds the
 *     fork-all/join-all seal against the call stream;
 *   - without a stream the call synchronizes every shard stream before
 *     returning (refused under `sync_policy::forbid`);
 *   - under CUDA graph capture the lane-ordered form requires the shard
 *     streams to be capturing already (`sharded_array::fork_from`), and
 *     refuses before enqueueing anything otherwise;
 *   - the body must not synchronize the host itself in the asynchronous
 *     form; if it needs a host-visible result, write it to a location
 *     (device slot, pinned staging) and `barrier(envs)` afterwards.
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

#include <cuda/std/__functional/invoke.h> // is_invocable_v
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/container/default_envs.cuh>
#include <cuda/experimental/__sharded/engine/visit_shards.cuh>

#include <cstddef>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
//! @brief Run @p body once per non-empty shard of @p data, on the shard's
//! environment's stream, under the sharded call contract (see file comment).
//!
//! @param data      any `sharded_view`
//! @param envs      one `sharded_env` per shard (`envs[g]` binds shard `g`)
//! @param body      host callable; `(g, d, envs[g])`, `(g, d, stream)` or
//!                  `(d, stream)` — it enqueues the shard's work and returns
//! @param call_env  per-call environment: stream ⇒ asynchronous
//!                  (lane-ordered, or bracketed if it says so); none ⇒
//!                  synchronous
_CCCL_TEMPLATE(class _S, class _Envs, class _Body, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void for_each_shard(_S&& data, const _Envs& envs, _Body body, const _CallEnv& call_env = {})
{
  __detail::__visit_shards(
    data, envs, call_env, "sharded::for_each_shard", [&](::std::size_t g, const auto& d, cudaStream_t s) {
      using __env_t = decltype(envs[g]);
      if constexpr (::cuda::std::is_invocable_v<_Body&, ::std::size_t, decltype(d), __env_t>)
      {
        body(g, d, envs[g]);
      }
      else if constexpr (::cuda::std::is_invocable_v<_Body&, ::std::size_t, decltype(d), cudaStream_t>)
      {
        body(g, d, s);
      }
      else
      {
        static_assert(::cuda::std::is_invocable_v<_Body&, decltype(d), cudaStream_t>,
                      "sharded::for_each_shard: body must be callable as (size_t, const shard&, const Env&), "
                      "(size_t, const shard&, cudaStream_t) or (const shard&, cudaStream_t)");
        body(d, s);
      }
    });
}

//! @brief Self-bound form: environments come from `default_envs(data)`.
_CCCL_TEMPLATE(class _S, class _Body, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_Body>>))
_CCCL_HOST_API void for_each_shard(_S&& data, _Body body, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  sharded::for_each_shard(::cuda::std::forward<_S>(data), envs, ::cuda::std::move(body), call_env);
}
} // namespace cuda::experimental::sharded
