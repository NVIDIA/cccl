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
 * @brief `stream_scope`: the stream's own context made current, for generic
 *        per-shard work.
 *
 * Work submitted into a CUDA stream executes in the stream's own context:
 * kernels launched into a stream created from a green context run on that
 * context's SM partition regardless of which context is current on the
 * calling thread (exercised by `cudax/test/places/stream_scope.cu`). What a
 * launch still needs from the calling thread is a current context matching
 * the stream's device; making the stream's exact context current also puts
 * everything context-sensitive done inside the scope (a kernel's first-launch
 * module load, function attributes, a handle created in place) where the
 * work runs.
 *
 * `stream_scope` is libcu++'s `cuda::__ensure_current_context` taken from the
 * stream (`cuStreamGetCtx_v2`, then a context push, popped on exit) — the
 * same scope the MGMN algorithms put around their per-rank calls — spelled
 * for a raw `cudaStream_t`. Generic algorithms over sharded structures
 * therefore never need an execution-place object: the per-shard environment's
 * stream carries everything.
 *
 * What deliberately stays outside this scope (provider/engine territory):
 * stream *creation* (streams must be born in their place's context — see
 * `stream_pool::next`), and long-lived context-implicit state such as vendor
 * library handles (create those under the owning place and cache them).
 *
 * Capture note: the stream's context query is capture-safe on CTK 13.4 (the
 * locality-domain feature set requires it anyway), so this scope is correct
 * while lanes are capturing, including cross-device captures.
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

#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/stream_ref.h> // defines __ensure_current_context(stream_ref)

#include <cuda_runtime.h>

namespace cuda::experimental::places
{
/**
 * @brief RAII scope making a stream's own context current for its lifetime
 * (green or primary), restoring the previous context on exit. A thin
 * spelling of `cuda::__ensure_current_context` for a raw `cudaStream_t`.
 * Non-copyable, non-movable.
 */
class stream_scope
{
public:
  explicit stream_scope(cudaStream_t __stream)
      : __ctx_{::cuda::stream_ref{__stream}}
  {}
  explicit stream_scope(::cuda::stream_ref __stream)
      : __ctx_{__stream}
  {}
  stream_scope(const stream_scope&)            = delete;
  stream_scope& operator=(const stream_scope&) = delete;
  stream_scope(stream_scope&&)                 = delete;
  stream_scope& operator=(stream_scope&&)      = delete;

private:
  ::cuda::__ensure_current_context __ctx_;
};
} // namespace cuda::experimental::places
