//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief A `stdexec`/P2300 scheduler for a single `exec_place`.
 *
 * `places::scheduler(place, res)` draws a stream from `place`'s pool (the
 * same pool `getStream` already draws from) and wraps it in an
 * `execution::stream_scheduler`. This is deliberately the simplest possible
 * bridge: it resolves the place to one concrete stream once, at the call
 * site, rather than deferring stream selection into `connect`/`start`. A
 * place is pool-based (`next()` may return a different stream each call);
 * this scheduler is not -- once obtained, it is a fixed, ordinary
 * `stream_scheduler` over whichever stream the pool happened to hand back.
 * Callers that want a fresh pool draw get one by calling `places::scheduler`
 * again, the same way they would call `getStream` again.
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

#include <cuda/experimental/__execution/stream/scheduler.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental::places
{
/**
 * @brief Return a `stream_scheduler` bound to a stream drawn from `place`'s
 * pool via `exec_place::getStream`.
 *
 * @param place The place to schedule work on.
 * @param res The stream-pool registry to draw from (same registry
 * `getStream`/`getDataStream` use).
 * @param for_computation Forwarded to `getStream`; selects the compute pool
 * (default) vs. the data pool.
 */
[[nodiscard]] inline execution::stream_scheduler
scheduler(const exec_place& place, exec_place_resources& res, bool for_computation = true)
{
  return execution::stream_scheduler{stream_ref{place.getStream(res, for_computation).stream}};
}

// An `async_resources_handle` overload is deliberately not provided here:
// `exec_place::getStream(async_resources_handle&, ...)` is only *declared*
// in `places.cuh` and defined inline in `async_resources_handle.cuh`, which
// this standalone `__places` header does not include (see
// `exec_place_resources.cuh`'s own note on the same tradeoff). Callers with
// a handle already in scope can pass `h.get_place_resources()` to the
// overload above.
} // namespace cuda::experimental::places
