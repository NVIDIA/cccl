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
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__fwd/devices.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

// Forward declare so we dont need CUDA 12.5 headers
struct CUgreenCtx_st;
using CUgreenCtx = ::CUgreenCtx_st*;

_CCCL_BEGIN_NAMESPACE_CUDA

class __logical_device_ref
{
public:
  enum class kinds : ::cuda::std::uint8_t
  {
    device,
    green_context
  };

  __logical_device_ref()                                   = delete;
  __logical_device_ref(device_ref, int)                    = delete;
  __logical_device_ref(device_ref, ::cuda::std::nullptr_t) = delete;

  _CCCL_HOST_API explicit __logical_device_ref(device_ref __dev)
      : __logical_device_ref{__dev, __dev.__primary_context()}
  {}

  _CCCL_HOST_API __logical_device_ref(device_ref __dev, ::CUgreenCtx __gctx)
      : __logical_device_ref{__dev,
#  if _CCCL_CTK_AT_LEAST(12, 5)
                             ::cuda::__driver::__ctxFromGreenCtx(__gctx),
#  else // ^^^ 12.5+ ^^^ / vvv 12.4- vvv
                             ::CUcontext{},
#  endif // ^^^ 12.4- ^^^
                             __gctx}
  {}

  _CCCL_HOST_API __logical_device_ref(device_ref __dev, ::CUcontext __ctx)
      : __logical_device_ref{__dev, __ctx, ::CUgreenCtx{}}
  {}

  [[nodiscard]] _CCCL_HOST_API device_ref underlying_device() const noexcept
  {
    return __device_;
  }

  [[nodiscard]] _CCCL_HOST_API kinds kind() const noexcept
  {
    return __green_ctx_ == nullptr ? kinds::device : kinds::green_context;
  }

  [[nodiscard]] _CCCL_HOST_API ::CUgreenCtx green_context() const noexcept
  {
    return __green_ctx_;
  }

  [[nodiscard]] _CCCL_HOST_API ::CUcontext context() const noexcept
  {
    return __cu_ctx_;
  }

  // Deliberately __ugly (and should stay __ugly after we publicize this class). The exact type
  // is an implementation detail, the only public properties of the class are the member names.
  //
  // Also use a bespoke structure instead of a pair to give more meaningful names to the
  // members instead of just "first" and "second".
  struct __locality_domain_result
  {
    ::cuda::std::size_t domain_id;
    bool localized;
  };

  [[nodiscard]] _CCCL_HOST_API __locality_domain_result locality_domain() const
  {
#  if _CCCL_CTK_AT_LEAST(13, 4)
    if (kind() == kinds::green_context)
    {
      const ::CUdevResource __sm_resource =
        ::cuda::__driver::__ctxGetDevResource(this->context(), ::CU_DEV_RESOURCE_TYPE_SM);

      if (__sm_resource.sm.flags & ::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID)
      {
        return {__sm_resource.sm.localityDomainId, /*localized=*/true};
      }
    }
#  endif // _CCCL_CTK_AT_LEAST(13, 4)
    // Either we have no localization at all (the device doesn't support it, or the CTK is too
    // old), or the green context itself isn't actually localized.
    //
    // In this case even if the green context exists, the locality domain is somewhat
    // meaningless, but represents the whole device since there is no guarantee where the
    // assigned SMs land. It is up to the caller to interpret the result.
    return {/*domain_id=*/0, /*localized=*/false};
  }

  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(const __logical_device_ref& __lhs, const __logical_device_ref& __rhs) noexcept
  {
    return __lhs.__device_ == __rhs.__device_ && __lhs.__cu_ctx_ == __rhs.__cu_ctx_
        && __lhs.__green_ctx_ == __rhs.__green_ctx_;
  }

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(const __logical_device_ref& __lhs, const __logical_device_ref& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#  endif // _CCCL_STD_VER <= 2017

protected:
  _CCCL_HOST_API constexpr __logical_device_ref(
    device_ref __device, ::CUcontext __cu_ctx, ::CUgreenCtx __green_ctx) noexcept
      : __device_{__device}
      , __cu_ctx_{__cu_ctx}
      , __green_ctx_{__green_ctx}
  {}

  device_ref __device_{0};
  ::CUcontext __cu_ctx_{};
  ::CUgreenCtx __green_ctx_{};
};

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_LOGICAL_DEVICE_REF_H
