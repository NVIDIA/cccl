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

#define CUDAX_GET_DRIVER_FUNCTION(function_name) \
  reinterpret_cast<decltype(function_name)*>(getDriverEntryPoint(#function_name))

namespace cuda::experimental::detail::driver
{
inline void* getDriverEntryPoint(const char* name)
{
  void* fn;
  cudaDriverEntryPointQueryResult result;
  cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &result);
  if (result != cudaDriverEntryPointSuccess) {
    if (result == cudaDriverEntryPointVersionNotSufficent) {
      ::cuda::__throw_cuda_error(cudaErrorNotSupported, "Driver does not support this API");
    }
    else {
      ::cuda::__throw_cuda_error(cudaErrorUnknown, "Failed to access driver API");
    }
  }
  return fn;
}

inline void ctxPush(CUcontext ctx)
{
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  CUresult status       = driver_fn(ctx);
  if (status != CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<cudaError_t>(status), "Failed to push context");
  }
}

inline void ctxPop()
{
  CUcontext dummy;
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  CUresult status       = driver_fn(&dummy);
  if (status != CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<cudaError_t>(status), "Failed to pop context");
  }
}

inline CUcontext streamGetCtx(CUstream stream)
{
  CUcontext result;
  static auto driver_fn = CUDAX_GET_DRIVER_FUNCTION(cuStreamGetCtx);
  CUresult status       = driver_fn(stream, &result);
  if (status != CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<cudaError_t>(status), "Failed to get context from a stream");
  }
  return result;
}
} // namespace cuda::experimental::detail::driver

#undef CUDAX_GET_DRIVER_FUNCTION
#endif