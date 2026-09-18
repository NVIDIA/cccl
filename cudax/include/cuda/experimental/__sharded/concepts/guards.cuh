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
 * @brief Entry guards shared by every sharded algorithm — the environment
 *        count check, the co-partitioning check, the `sync_policy::forbid`
 *        guard (`require_sync_allowed`) and the capture guards — plus the
 *        stream plumbing they and the engines share: `stream_scope`
 *        (`places::stream_scope`, made available in this namespace) and
 *        `__detail::__wait_stream_on`.
 *
 * Every guard runs before anything is enqueued, so a refusal leaves the
 * caller's state (and any open capture) valid.
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

#include <cuda/experimental/__places/place_group.cuh> // check_not_capturing
#include <cuda/experimental/__places/stream_scope.cuh>
#include <cuda/experimental/__sharded/concepts/env.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::sharded
{
//! @brief RAII device scope derived from a stream (see
//! `<cuda/experimental/__places/stream_scope.cuh>`).
using ::cuda::experimental::places::stream_scope;

//! @brief Guard for the `sync_policy::forbid` contract: throw before a
//! would-be host synchronization, leaving all state valid.
//!
//! Every internal host-blocking site of the algorithm tier routes through
//! this (the `check_not_capturing` discipline, generalized). Amortized state
//! warm-up (handle/plan creation) is exempt by contract: warm up before
//! entering a no-sync region.
template <class _CallEnv>
void require_sync_allowed(const _CallEnv& __env, const char* __what)
{
  if (query_sync_policy(__env) == sync_policy::forbid)
  {
    throw ::std::runtime_error(
      ::std::string(__what)
      + ": operation would synchronize with the host, but the call "
        "environment carries sync_policy::forbid");
  }
}

namespace reserved
{
//! @brief Entry guard shared by every algorithm: a `sharded_env_range` must
//! carry exactly one environment per shard. Extra environments are refused
//! too, since they almost always mean a mismatched view/envs pairing.
//! @throws std::invalid_argument prefixed with @p __what.
template <class _Envs>
void __check_env_count(const _Envs& __envs, ::std::size_t __num_shards, const char* __what)
{
  if (__env_count(__envs) != __num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": environment count does not match shard count");
  }
}

//! @brief Check that two sharded views are co-partitioned: same shard count
//! and, per shard, identical global regions.
template <class _SA, class _SB>
void __check_copartitioned(const _SA& __a, const _SB& __b, const char* __what)
{
  const ::std::size_t __n = __shard_count(__a);
  if (__n != __shard_count(__b))
  {
    throw ::std::invalid_argument(::std::string(__what) + ": shard count mismatch");
  }
  for (::std::size_t __g = 0; __g < __n; ++__g)
  {
    if (static_cast<::std::size_t>(__a.shard(__g).size) != static_cast<::std::size_t>(__b.shard(__g).size)
        || static_cast<::std::size_t>(__a.shard(__g).global_offset)
             != static_cast<::std::size_t>(__b.shard(__g).global_offset))
    {
      throw ::std::invalid_argument(::std::string(__what) + ": shard regions differ (not co-partitioned)");
    }
  }
}

//! @brief Capture entry guard for synchronous / host-side operations: refuse
//! when a global-mode capture is open anywhere in the process (legacy-stream
//! probe) or when any of the @p __count streams named by @p __stream_at is
//! itself being captured (null streams are skipped). Runs before anything is
//! enqueued, so a refusal leaves the caller's capture valid. Each probe is a
//! ~40 ns host-side query.
template <class _StreamAt>
void __check_not_capturing_all(::std::size_t __count, const char* __what, _StreamAt __stream_at)
{
  places::check_not_capturing(nullptr, __what);
  for (::std::size_t __i = 0; __i < __count; ++__i)
  {
    if (const cudaStream_t __s = __stream_at(__i))
    {
      places::check_not_capturing(__s, __what);
    }
  }
}

//! @brief `__check_not_capturing_all` over the first @p __num_shards streams
//! of a `sharded_env_range`.
template <class _Envs>
void __check_envs_not_capturing(const _Envs& __envs, ::std::size_t __num_shards, const char* __what)
{
  __check_not_capturing_all(__num_shards, __what, [&](::std::size_t __g) {
    return ::cuda::get_stream(__envs[__g]).get();
  });
}

//! @brief `__check_not_capturing_all` over a single (possibly null) stream.
inline void __check_stream_not_capturing(cudaStream_t __stream, const char* __what)
{
  __check_not_capturing_all(1, __what, [&](::std::size_t) {
    return __stream;
  });
}
} // namespace reserved

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
} // namespace __detail
} // namespace cuda::experimental::sharded

// NOLINTEND(bugprone-reserved-identifier)
