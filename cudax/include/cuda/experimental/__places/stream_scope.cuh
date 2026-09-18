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
 * model, and is exercised by `cudax/test/places/stream_scope.cu`). The one
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

#include <cuda/experimental/__places/places.cuh> // cuda_try
#include <cuda/experimental/__places/stream_pool.cuh> // get_device_from_stream

#include <cuda_runtime.h>

namespace cuda::experimental::places
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
} // namespace cuda::experimental::places
