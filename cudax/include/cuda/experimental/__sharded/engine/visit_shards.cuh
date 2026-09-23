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
 * @brief `__detail::__visit_shards`: the concept-tier shard visitor — the
 *        shared driver of every algorithm whose engine is a per-shard launch
 *        (the map family, and per-shard kernels with a halo edge such as
 *        `adjacent_difference`). No cross-shard combine stage lives here;
 *        a body may add its own lane edges (`__wait_stream_on`) before its
 *        launch.
 *
 * Visits every non-empty shard under `stream_scope` on its environment's
 * stream and applies the sharded call contract around the per-shard body:
 * the environment-count guard, the synchronous no-stream form (refused under
 * `sync_policy::forbid`), lane-ordered or `composition::bracketed`
 * composition, and the capture-time refusal. Algorithms whose engine is a
 * per-shard launch (`transform`, `fill`, `segmented_reduce`, ...) go through
 * it; algorithms driven by an MGMN engine go through `engine/mgmn.cuh`.
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
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__functional/invoke.h> // is_invocable_v

#include <cuda/experimental/__places/place_group.cuh> // stream_in_capture
#include <cuda/experimental/__sharded/concepts/env.cuh>
#include <cuda/experimental/__sharded/concepts/guards.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::sharded
{
namespace __detail
{
//! @brief The concept-tier shard visitor (successor of the container's
//! `each_shard`, with environments as the stream source and the call
//! contract enforced): visit every non-empty shard under `stream_scope` on its
//! environment's stream, with the per-call environment selecting the
//! contract — stream present = asynchronous (LANE-ORDERED by default: the
//! call enqueues each shard's work on its environment's stream and touches
//! nothing else — consecutive calls on the same environments are ordered
//! per lane by stream order, independent across lanes; zero host
//! synchronization; a call environment carrying `composition::bracketed`
//! restores the per-call fork-all/join-all seal against the call stream),
//! no stream = synchronous convenience (refused under
//! `sync_policy::forbid`).
//!
//! Lane-ordered calls under CUDA graph capture require the environments'
//! streams to be capturing already (the caller forks the lanes from the
//! capture origin once per pipeline — `sharded_array::fork_from`, or entry
//! edges of their own); a lane-ordered call whose call stream is capturing
//! while a shard stream is not is REFUSED before any work is enqueued —
//! the work would silently escape the graph otherwise.
//!
//! @p __body is a host callable `(const descriptor&, cudaStream_t)` — or,
//! for algorithms that need the shard index (cross-shard boundary logic),
//! `(size_t, const descriptor&, cudaStream_t)` — that enqueues the shard's
//! work on the given stream (the `each_shard` dual-arity convention).
template <class _S, class _Envs, class _CallEnv, class _PerShard>
_CCCL_HOST_API void
__visit_shards(_S&& __data, const _Envs& __envs, const _CallEnv& __call_env, const char* __what, _PerShard __body)
{
  const ::std::size_t __num_shards = reserved::__shard_count(__data);
  reserved::__check_env_count(__envs, __num_shards, __what);

  constexpr bool __is_async         = async_call_env<_CallEnv>;
  [[maybe_unused]] bool __bracketed = false;

  if constexpr (!__is_async)
  {
    // Refusals first, before any CUDA call: this form synchronizes at the
    // end, so both refusal conditions must be decided before any work is
    // enqueued (the entry-guard discipline, applied family-wide).
    require_sync_allowed(__call_env, __what);
    reserved::__check_envs_not_capturing(__envs, __num_shards, __what);
  }
  else
  {
    __bracketed = query_composition(__call_env) == composition::bracketed;
    if (!__bracketed && places::stream_in_capture(::cuda::get_stream(__call_env).get()))
    {
      // Lane-ordered under capture: every lane must already be part of the
      // capture, or its work would silently escape the graph. Refused at
      // entry, before anything is enqueued (the capture stays valid).
      for (const auto __g : each(__num_shards))
      {
        if (__data.shard(__g).size != 0 && !places::stream_in_capture(::cuda::get_stream(__envs[__g]).get()))
        {
          _CCCL_THROW(::std::runtime_error,
                      ::std::string(__what)
                        + ": lane-ordered asynchronous call during CUDA graph capture requires the "
                          "shard environments' streams to be capturing (fork the lanes from the "
                          "capture stream once per pipeline, e.g. sharded_array::fork_from), or opt "
                          "into composition::bracketed on the call environment");
        }
      }
    }
  }

  for (const auto __g : each(__num_shards))
  {
    const auto& __d = __data.shard(__g);
    if (__d.size == 0)
    {
      continue;
    }
    const ::cuda::stream_ref __shard_stream = ::cuda::get_stream(__envs[__g]);
    if constexpr (__is_async)
    {
      if (__bracketed)
      {
        __wait_stream_on(__shard_stream.get(), ::cuda::get_stream(__call_env).get());
      }
    }
    stream_scope __scope(__shard_stream.get());
    if constexpr (::cuda::std::is_invocable_v<_PerShard&, ::std::size_t, decltype(__d), cudaStream_t>)
    {
      __body(__g, __d, __shard_stream.get());
    }
    else
    {
      __body(__d, __shard_stream.get());
    }
  }

  if constexpr (__is_async)
  {
    if (__bracketed)
    {
      for (const auto __g : each(__num_shards))
      {
        if (__data.shard(__g).size != 0)
        {
          __wait_stream_on(::cuda::get_stream(__call_env).get(), ::cuda::get_stream(__envs[__g]).get());
        }
      }
    }
  }
  else
  {
    for (const auto __g : each(__num_shards))
    {
      if (__data.shard(__g).size != 0)
      {
        ::cuda::experimental::stf::cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(__envs[__g]).get()));
      }
    }
  }
}
} // namespace __detail
} // namespace cuda::experimental::sharded

// NOLINTEND(bugprone-reserved-identifier)
