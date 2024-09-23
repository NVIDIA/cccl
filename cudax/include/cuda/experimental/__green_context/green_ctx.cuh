//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GREEN_CONTEXT_GREEN_CTX
#define _CUDAX__GREEN_CONTEXT_GREEN_CTX

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/utility>

#include <cuda/experimental/__device/all_devices.cuh>
#include <cuda/experimental/__utility/driver_api.cuh>

#if CUDART_VERSION >= 12050
namespace cuda::experimental
{
struct device_ref;

struct green_context
{
  int __dev_id            = -1;
  CUgreenCtx __green_ctx  = nullptr;
  CUcontext __transformed = nullptr;

  explicit green_context(const device& __device)
      : __dev_id(__device.get())
  {
    // TODO get CUdevice from device
    auto __dev_handle = detail::driver::deviceGet(__dev_id);
    __green_ctx       = detail::driver::greenCtxCreate(__dev_handle);
    __transformed     = detail::driver::ctxFromGreenCtx(__green_ctx);
  }

  green_context(const green_context&)            = delete;
  green_context& operator=(const green_context&) = delete;

  // TODO this probably should be the runtime equivalent once available
  _CCCL_NODISCARD static green_context from_native_handle(CUgreenCtx __gctx)
  {
    int __id;
    CUcontext __transformed = detail::driver::ctxFromGreenCtx(__gctx);
    detail::driver::ctxPush(__transformed);
    _CCCL_TRY_CUDA_API(cudaGetDevice, "Failed to get device ordinal from a green context", &__id);
    detail::driver::ctxPop();
    return green_context(__id, __gctx, __transformed);
  }

  _CCCL_NODISCARD CUgreenCtx release() noexcept
  {
    __transformed = nullptr;
    __dev_id      = -1;
    return _CUDA_VSTD::exchange(__green_ctx, nullptr);
  }

  ~green_context()
  {
    if (__green_ctx)
    {
      [[maybe_unused]] cudaError_t __status = detail::driver::greenCtxDestroy(__green_ctx);
    }
  }

private:
  explicit green_context(int __id, CUgreenCtx __gctx, CUcontext __ctx)
      : __dev_id(__id)
      , __green_ctx(__gctx)
      , __transformed(__ctx)
  {}
};
} // namespace cuda::experimental
#endif // CUDART_VERSION >= 12050
#endif // _CUDAX__GREEN_CONTEXT_GREEN_CTX
