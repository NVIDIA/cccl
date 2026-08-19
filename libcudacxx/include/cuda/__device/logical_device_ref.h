//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_LOGICAL_DEVICE_REF_H
#define _CUDA___DEVICE_LOGICAL_DEVICE_REF_H

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

#  include <cuda/std/__cccl/prologue.h>

// Forward declare so we dont need CUDA 12.5 headers
struct CUgreenCtx_st;
using CUgreenCtx = ::CUgreenCtx_st*;

_CCCL_BEGIN_NAMESPACE_CUDA

class __logical_device_ref
{
public:
  __logical_device_ref() = delete;

  _CCCL_HOST_API constexpr __logical_device_ref(device_ref __dev, ::CUgreenCtx __gctx) noexcept
      : __device_{__dev}
      , __gctx_{__gctx}
  {}

  [[nodiscard]] _CCCL_HOST_API constexpr device_ref underlying_device() const noexcept
  {
    return __device_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::CUgreenCtx green_context() const noexcept
  {
    return __gctx_;
  }

  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(const __logical_device_ref& __lhs, const __logical_device_ref& __rhs) noexcept
  {
    return __lhs.__device_ == __rhs.__device_ && __lhs.__gctx_ == __rhs.__gctx_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(const __logical_device_ref& __lhs, const __logical_device_ref& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#  endif // _CCCL_STD_VER <= 2017

protected:
  device_ref __device_{0};
  ::CUgreenCtx __gctx_{};
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_LOGICAL_DEVICE_REF_H
