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
 * @brief Event pool backing the containers' `fork_from` / `join_into` stream
 *        ordering members.
 *
 * Design note (ownership of the events). The pool lives INSIDE each container
 * rather than in a `place_group` resource cache: shards do not hold a
 * reference to the group that created their streams, and ADOPTED containers
 * (`sharded_array<T>::adopt`, views over foreign streams) never had one. A
 * group-owned cache would either force the container to carry a group
 * lifetime dependency or leave the adopted path on per-call transient events.
 * Container-owned events are correct for every construction path; the cost is
 * at most one event per shard plus one per caller-stream device, created
 * lazily and reused for the container's lifetime.
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/std/cstddef>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/places.cuh>

#include <mutex>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded::reserved
{
/**
 * @brief Lazily created, reusable `cudaEventDisableTiming` events for stream
 *        fork/join choreography.
 *
 * Two kinds of slots:
 *  - JOIN events, one per shard index, created with the shard's execution
 *    place active (so the event lives in the context that owns the shard's
 *    stream);
 *  - FORK events, one per caller-stream device (an event must be created on
 *    the device of the stream it is recorded on; caller streams may come from
 *    any device, so they are keyed by device ordinal).
 *
 * Lazy creation is mutex-guarded (thread-safe). RECORDING on a pooled event
 * is not serialized here: concurrent `fork_from`/`join_into` calls on the
 * same container would race on the shared events and must be ordered
 * externally — same contract as the containers' other stream-ordered members.
 *
 * All operations are capture-safe: event creation is not a stream operation,
 * and record/wait inside an active capture become graph dependencies.
 */
class fork_join_event_pool
{
public:
  fork_join_event_pool() = default;

  fork_join_event_pool(fork_join_event_pool&& other) noexcept
      : join_events_(::std::move(other.join_events_))
      , fork_events_(::std::move(other.fork_events_))
  {
    other.join_events_.clear();
    other.fork_events_.clear();
  }

  fork_join_event_pool& operator=(fork_join_event_pool&& other) noexcept
  {
    if (this != &other)
    {
      destroy();
      join_events_ = ::std::move(other.join_events_);
      fork_events_ = ::std::move(other.fork_events_);
      other.join_events_.clear();
      other.fork_events_.clear();
    }
    return *this;
  }

  fork_join_event_pool(const fork_join_event_pool&)            = delete;
  fork_join_event_pool& operator=(const fork_join_event_pool&) = delete;

  ~fork_join_event_pool()
  {
    destroy();
  }

  /**
   * @brief The join event for shard @p idx, created on first use.
   *
   * The caller must have the shard's execution place ACTIVE (the event is
   * created in the current context so it matches the shard's stream).
   */
  cudaEvent_t join_event(::cuda::std::size_t idx)
  {
    const ::std::lock_guard<::std::mutex> lock(mutex_);
    if (idx >= join_events_.size())
    {
      join_events_.resize(idx + 1, nullptr);
    }
    if (!join_events_[idx])
    {
      places::cuda_safe_call(cudaEventCreateWithFlags(&join_events_[idx], cudaEventDisableTiming));
    }
    return join_events_[idx];
  }

  /**
   * @brief The fork event for caller streams on device @p device, created on
   *        first use (with @p device temporarily made current).
   */
  cudaEvent_t fork_event(int device)
  {
    const ::std::lock_guard<::std::mutex> lock(mutex_);
    for (const auto& [dev, ev] : fork_events_)
    {
      if (dev == device)
      {
        return ev;
      }
    }

    int prev = -1;
    places::cuda_safe_call(cudaGetDevice(&prev));
    if (prev != device)
    {
      places::cuda_safe_call(cudaSetDevice(device));
    }
    cudaEvent_t ev                = nullptr;
    const cudaError_t create_stat = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    // Restore the current device on EVERY path before surfacing a failure.
    if (prev != device)
    {
      places::cuda_safe_call(cudaSetDevice(prev));
    }
    places::cuda_safe_call(create_stat);
    fork_events_.emplace_back(device, ev);
    return ev;
  }

private:
  void destroy() noexcept
  {
    // cudaEventDestroy on an in-flight event is safe: destruction is deferred
    // by the driver until the event completes.
    for (auto ev : join_events_)
    {
      if (ev)
      {
        cudaEventDestroy(ev);
      }
    }
    join_events_.clear();
    for (const auto& [dev, ev] : fork_events_)
    {
      cudaEventDestroy(ev);
    }
    fork_events_.clear();
  }

  ::std::mutex mutex_;
  ::std::vector<cudaEvent_t> join_events_; // one per shard index
  ::std::vector<::std::pair<int, cudaEvent_t>> fork_events_; // one per caller-stream device
};
} // namespace cuda::experimental::sharded::reserved
