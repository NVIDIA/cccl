//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_LOGICAL_DEVICE_H
#define _CUDA___DEVICE_LOGICAL_DEVICE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__device/logical_device_ref.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

// Forward declare so we dont need CUDA 12.5 headers
struct CUgreenCtx_st;
using CUgreenCtx = ::CUgreenCtx_st*;

_CCCL_BEGIN_NAMESPACE_CUDA

class __logical_device : public __logical_device_ref
{
public:
  _CCCL_HOST_API explicit __logical_device(device_ref __device)
      : __logical_device_ref{__device}
  {}

  [[nodiscard]] _CCCL_HOST_API static __logical_device from_native_handle(device_ref __device, ::CUgreenCtx __gctx)
  {
    return __logical_device{__device, __gctx};
  }

  static __logical_device from_native_handle(device_ref, int)                    = delete;
  static __logical_device from_native_handle(device_ref, ::cuda::std::nullptr_t) = delete;

  // Must use from_native_handle() for now
  __logical_device()                                   = delete;
  __logical_device(const __logical_device&)            = delete;
  __logical_device& operator=(const __logical_device&) = delete;

  _CCCL_HOST_API __logical_device(__logical_device&& __other) noexcept
      : __logical_device_ref{::cuda::std::move(__other.__device_),
                             ::cuda::std::exchange(__other.__cu_ctx_, nullptr),
                             ::cuda::std::exchange(__other.__green_ctx_, nullptr)}
  {}

  _CCCL_HOST_API __logical_device& operator=(__logical_device&& __other) noexcept
  {
    if (this != &__other)
    {
      __reset();
      __device_    = ::cuda::std::move(__other.__device_);
      __cu_ctx_    = ::cuda::std::exchange(__other.__cu_ctx_, nullptr);
      __green_ctx_ = ::cuda::std::exchange(__other.__green_ctx_, nullptr);
    }
    return *this;
  }

  _CCCL_HOST_API ~__logical_device()
  {
    __reset();
  }

private:
  _CCCL_HOST_API explicit __logical_device(device_ref __device, ::CUgreenCtx __gctx)
      : __logical_device_ref{__device, __gctx}
  {}

  _CCCL_HOST_API void __reset() noexcept
  {
    if (this->kind() == kinds::green_context)
    {
#  if _CCCL_CTK_AT_LEAST(12, 5)
      _CCCL_ASSERT_DRIVER_API(
        ::cuda::__driver::__greenCtxDestroyNoThrow, "Failed to destroy green context", green_context());
#  endif // _CCCL_CTK_AT_LEAST(12, 5)
      __green_ctx_ = nullptr;
      __cu_ctx_    = nullptr;
    }
  }
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_LOGICAL_DEVICE_H
