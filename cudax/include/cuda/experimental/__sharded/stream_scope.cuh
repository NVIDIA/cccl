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
 * @brief `stream_scope`: device currency derived from a stream, for generic
 *        per-shard work.
 *
 * Work submitted into a CUDA stream executes in the stream's own context:
 * kernels launched into a stream created from a green context run on that
 * context's SM partition regardless of which context is current on the
 * calling thread (this is the design premise of the runtime execution-context
 * model, and is exercised by `cudax/test/sharded/stream_scope.cu`). The one
 * thing a launch still needs from the calling thread is *device* currency —
 * the runtime requires the current device to match the stream's device.
 *
 * `stream_scope` provides exactly that, derived from the stream alone:
 * `cudaSetDevice(get_device_from_stream(stream))` with RAII restore. Generic
 * algorithms over sharded structures therefore never need an execution-place
 * object: the per-shard environment's stream carries everything.
 *
 * What deliberately stays outside this scope (provider/engine territory):
 * stream *creation* (streams must be born in their place's context — see
 * `stream_pool::next`), and context-implicit state creation such as vendor
 * library handles (create those under the owning place, before entering
 * generic code, and cache them).
 *
 * Capture note: the device query is capture-safe. On CTK >= 12.8,
 * `cudaStreamGetDevice` answers during thread-local, relaxed and global
 * capture, on device and green-context streams, without invalidating the
 * capture (probed on CTK 13.4) — so this scope is correct while lanes are
 * capturing, including cross-device captures. (Pre-12.8 toolkits fall back
 * to the calling thread's current device under capture, which is only
 * correct on single-device systems; the locality-domain feature set
 * requires 13.4+ anyway.)
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

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__functional/invoke.h> // is_invocable_v

#include <cuda/experimental/__places/place_group.cuh> // check_not_capturing
#include <cuda/experimental/__places/stream_pool.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief RAII device scope derived from a stream: makes the stream's device
 * current for the scope's lifetime and restores the previous device on exit.
 *
 * On a single-device system (or when the stream's device is already current)
 * this is a no-op apart from one device query. Non-copyable, non-movable.
 */
class stream_scope
{
public:
  explicit stream_scope(cudaStream_t __stream)
  {
    const int __target = places::get_device_from_stream(__stream);
    __prev_            = places::cuda_try<cudaGetDevice>();
    if (__target != __prev_)
    {
      ::cuda::experimental::stf::cuda_safe_call(cudaSetDevice(__target));
      __switched_ = true;
    }
  }

  explicit stream_scope(::cuda::stream_ref __stream)
      : stream_scope(__stream.get())
  {}

  stream_scope(const stream_scope&)            = delete;
  stream_scope& operator=(const stream_scope&) = delete;
  stream_scope(stream_scope&&)                 = delete;
  stream_scope& operator=(stream_scope&&)      = delete;

  ~stream_scope()
  {
    if (__switched_)
    {
      // Restore on every path; a failure here would indicate a torn-down
      // context, in which case there is nothing better to do than continue.
      (void) cudaSetDevice(__prev_);
    }
  }

private:
  int __prev_      = -1;
  bool __switched_ = false;
};

namespace __detail
{
//! @brief Make @p __consumer wait for all work currently enqueued on
//! @p __producer (transient-event idiom; capture-legal: record/wait become
//! graph dependencies).
//!
//! The event is created under the producer stream's device (events must be
//! created where they are recorded; cross-device stream waits are legal) and
//! destroyed immediately after the wait is enqueued — the driver defers the
//! release until completion.
inline void __wait_stream_on(cudaStream_t __consumer, cudaStream_t __producer)
{
  if (__consumer == __producer)
  {
    return;
  }
  stream_scope __scope(__producer);
  cudaEvent_t __ev = nullptr;
  ::cuda::experimental::stf::cuda_safe_call(cudaEventCreateWithFlags(&__ev, cudaEventDisableTiming));
  ::cuda::experimental::stf::cuda_safe_call(cudaEventRecord(__ev, __producer));
  ::cuda::experimental::stf::cuda_safe_call(cudaStreamWaitEvent(__consumer, __ev, 0));
  ::cuda::experimental::stf::cuda_safe_call(cudaEventDestroy(__ev));
}

//! @brief Shared driver for the concept-generic map family (no cross-shard
//! stage): visit every non-empty shard under `stream_scope` on its
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
__generic_map(_S&& __data, const _Envs& __envs, const _CallEnv& __call_env, const char* __what, _PerShard __body)
{
  const ::std::size_t __num_shards = reserved::__shard_count(__data);
  if (reserved::__env_count(__envs) < __num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": fewer environments than shards");
  }

  constexpr bool __is_async         = async_call_env<_CallEnv>;
  [[maybe_unused]] bool __bracketed = false;

  if constexpr (!__is_async)
  {
    // Refusals first, before any CUDA call: this form synchronizes at the
    // end, so both refusal conditions must be decided before any work is
    // enqueued (the entry-guard discipline, applied family-wide).
    require_sync_allowed(__call_env, __what);
    places::check_not_capturing(nullptr, __what);
    for (const auto __g : each(__num_shards))
    {
      places::check_not_capturing(::cuda::get_stream(__envs[__g]).get(), __what);
    }
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
