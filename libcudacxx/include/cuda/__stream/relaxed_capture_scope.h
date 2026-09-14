//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___STREAM_RELAXED_CAPTURE_SCOPE_H
#define _CUDA___STREAM_RELAXED_CAPTURE_SCOPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/__exception/exception_macros.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief RAII scope putting the calling thread in relaxed stream-capture mode.
//!
//! Some driver calls are "potentially unsafe" under an active stream capture and invalidate the
//! capture (or fail with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`) when the calling thread is in
//! global or thread-local capture mode: memory-pool attribute reads and writes are among them.
//! Capture mode is a per-thread property, so switching THIS thread to relaxed mode for the
//! duration of such a call lets it execute immediately (it is not recorded into the graph) and
//! leaves the capture valid. Nothing that should be captured may be issued inside the scope.
//!
//! There is no query for a thread's capture mode; the exchange is the only per-thread primitive,
//! and it has no observable effect when the thread is not capturing, so the scope needs no
//! capture check and costs two cheap driver calls.
struct [[maybe_unused]] __relaxed_capture_scope
{
  _CCCL_HOST_API __relaxed_capture_scope()
      : __previous_{::CU_STREAM_CAPTURE_MODE_RELAXED}
  {
    ::cuda::__driver::__threadExchangeStreamCaptureMode(__previous_);
  }

  _CCCL_HOST_API ~__relaxed_capture_scope() noexcept
  {
    // Restore whatever the thread had. The exchange cannot fail for a mode it returned itself, so
    // the status is deliberately dropped rather than caught.
    (void) ::cuda::__driver::__threadExchangeStreamCaptureModeNoThrow(__previous_);
  }

  __relaxed_capture_scope(const __relaxed_capture_scope&)            = delete;
  __relaxed_capture_scope& operator=(const __relaxed_capture_scope&) = delete;

private:
  ::CUstreamCaptureMode __previous_;
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_RELAXED_CAPTURE_SCOPE_H
