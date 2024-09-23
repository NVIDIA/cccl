//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_DRIVER_API
#define _CUDAX__UTILITY_DRIVER_API

#include <cuda.h>

#include <cuda/std/__exception/cuda_error.h>

// Get the driver function by name using this macro
#define CUDAX_GET_DRIVER_FUNCTION(function_name) \
  reinterpret_cast<decltype(function_name)*>(get_driver_entry_point(#function_name))

#define CUDAX_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, version) \
  reinterpret_cast<decltype(versioned_fn_name)*>(get_driver_entry_point(#function_name, version))

namespace cuda::experimental::detail::driver
{
//! @brief Get a driver function pointer for a given API name and optionally specific CUDA version
//!
//! For minor version compatibility request the 12.0 version of everything for now, unless requested otherwise
inline void* get_driver_entry_point(const char* name, [[maybe_unused]] int version = 12000)
{
  void* fn;
  cudaDriverEntryPointQueryResult result;
#if CUDART_VERSION >= 12050
  cudaGetDriverEntryPointByVersion(name, &fn, version, cudaEnableDefault, &result);
#else
  // Versioned get entry point not available before 12.5, but we don't need anything versioned before that
  cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &result);
#endif
  if (result != cudaDriverEntryPointSuccess)
  {
    if (result == cudaDriverEntryPointVersionNotSufficent)
    {
      ::cuda::__throw_cuda_error(cudaErrorNotSupported, "Driver does not support this API");
    }
    else
    {
      ::cuda::__throw_cuda_error(cudaErrorUnknown, "Failed to access driver API");
    }
  }
  return fn;
}

template <typename Fn, typename... Args>
inline void call_driver_fn(Fn fn, const char* err_msg, Args... args)
{
  CUresult status = fn(args...);
  if (status != CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<cudaError_t>(status), err_msg);
  }
}

inline int getVersion()
{
  static int version = []() {
    int v;
    auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuDriverGetVersion);
    call_driver_fn(driver_fn, "Failed to check CUDA driver version", &v);
    return v;
  }();
  return version;
}

inline void ctxPush(CUcontext ctx)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  call_driver_fn(driver_fn, "Failed to push context", ctx);
}

inline CUcontext ctxPop()
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  CUcontext result;
  call_driver_fn(driver_fn, "Failed to pop context", &result);
  return result;
}

inline CUcontext ctxGetCurrent()
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxGetCurrent);
  CUcontext result;
  call_driver_fn(driver_fn, "Failed to get current context", &result);
  return result;
}

inline CUdevice deviceGet(int ordinal)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuDeviceGet);
  CUdevice result;
  call_driver_fn(driver_fn, "Failed to get device", &result, ordinal);
  return result;
}

inline CUcontext primaryCtxRetain(CUdevice dev)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRetain);
  CUcontext result;
  call_driver_fn(driver_fn, "Failed to retain context for a device", &result, dev);
  return result;
}

inline void primaryCtxRelease(CUdevice dev)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRelease);
  // TODO we might need to ignore failure here
  call_driver_fn(driver_fn, "Failed to release context for a device", dev);
}

inline bool isPrimaryCtxActive(CUdevice dev)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxGetState);
  int result;
  unsigned int dummy;
  call_driver_fn(driver_fn, "Failed to check the primary ctx state", dev, &dummy, &result);
  return result == 1;
}

inline CUcontext streamGetCtx(CUstream stream)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuStreamGetCtx);
  CUcontext result;
  call_driver_fn(driver_fn, "Failed to get context from a stream", stream, &result);
  return result;
}

#if CUDART_VERSION >= 12050
struct __ctx_from_stream
{
  enum class __kind
  {
    __device,
    __green
  };

  __kind __ctx_kind;
  union
  {
    CUcontext __device;
    CUgreenCtx __green;
  } __ctx_ptr;
};

inline __ctx_from_stream streamGetCtx_v2(CUstream stream)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetCtx, cuStreamGetCtx_v2, 12050);
  CUcontext ctx;
  CUgreenCtx gctx;
  __ctx_from_stream __result;
  call_driver_fn(driver_fn, "Failed to get context from a stream", stream, &ctx, &gctx);
  if (gctx)
  {
    __result.__ctx_kind        = __ctx_from_stream::__kind::__green;
    __result.__ctx_ptr.__green = gctx;
  }
  else
  {
    __result.__ctx_kind         = __ctx_from_stream::__kind::__device;
    __result.__ctx_ptr.__device = ctx;
  }
  return __result;
}
#endif // CUDART_VERSION >= 12050

inline void streamWaitEvent(CUstream stream, CUevent event)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuStreamWaitEvent);
  call_driver_fn(driver_fn, "Failed to make a stream wait for an event", stream, event, CU_EVENT_WAIT_DEFAULT);
}

inline void eventRecord(CUevent event, CUstream stream)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuEventRecord);
  call_driver_fn(driver_fn, "Failed to record CUDA event", event, stream);
}

// Destroy calls return error codes to let the calling code decide if the error should be ignored
inline cudaError_t streamDestroy(CUstream stream)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuStreamDestroy);
  return static_cast<cudaError_t>(driver_fn(stream));
}

inline cudaError_t eventDestroy(CUevent event)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuEventDestroy);
  return static_cast<cudaError_t>(driver_fn(event));
}

#if CUDART_VERSION >= 12050
// Add actual resource description input once exposure is ready
inline CUgreenCtx greenCtxCreate(CUdevice dev)
{
  CUgreenCtx result;
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, cuGreenCtxCreate, 12050);
  call_driver_fn(driver_fn, "Failed to create a green context", &result, nullptr, dev, CU_GREEN_CTX_DEFAULT_STREAM);
  return result;
}

inline cudaError_t greenCtxDestroy(CUgreenCtx green_ctx)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxDestroy, cuGreenCtxDestroy, 12050);
  return static_cast<cudaError_t>(driver_fn(green_ctx));
}

inline CUcontext ctxFromGreenCtx(CUgreenCtx green_ctx)
{
  CUcontext result;
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION_VERSIONED(cuCtxFromGreenCtx, cuCtxFromGreenCtx, 12050);
  call_driver_fn(driver_fn, "Failed to convert a green context", &result, green_ctx);
  return result;
}
#endif // CUDART_VERSION >= 12050
} // namespace cuda::experimental::detail::driver

#undef CUDAX_GET_DRIVER_FUNCTION
#endif
