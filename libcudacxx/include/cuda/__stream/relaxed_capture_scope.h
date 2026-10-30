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

//! @brief RAII scope putting the calling thread in relaxed stream-capture mode, for driver calls
//! that are refused under global/thread-local capture. Nothing to be captured may be issued inside.
struct [[maybe_unused]] __relaxed_capture_scope
{
  _CCCL_HOST_API __relaxed_capture_scope()
  {
    ::cuda::__driver::__threadExchangeStreamCaptureMode(__previous_);
  }

  _CCCL_HOST_API ~__relaxed_capture_scope() noexcept
  {
    // Cannot fail for a mode the exchange itself returned.
    (void) ::cuda::__driver::__threadExchangeStreamCaptureModeNoThrow(__previous_);
  }

  __relaxed_capture_scope(const __relaxed_capture_scope&)            = delete;
  __relaxed_capture_scope& operator=(const __relaxed_capture_scope&) = delete;

private:
  ::CUstreamCaptureMode __previous_{::CU_STREAM_CAPTURE_MODE_RELAXED};
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___STREAM_RELAXED_CAPTURE_SCOPE_H
