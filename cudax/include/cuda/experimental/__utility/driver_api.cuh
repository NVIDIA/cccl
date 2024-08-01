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

namespace cuda::experimental::detail::driver
{
inline void* get_driver_entry_point(const char* name)
{
  void* fn;
  cudaDriverEntryPointQueryResult result;
  cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &result);
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

inline void ctxPush(CUcontext ctx)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  call_driver_fn(driver_fn, "Failed to push context", ctx);
}

inline void ctxPop()
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  CUcontext dummy;
  call_driver_fn(driver_fn, "Failed to pop context", &dummy);
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
} // namespace cuda::experimental::detail::driver

#undef CUDAX_GET_DRIVER_FUNCTION
#endif
