//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__STD__CUDA_API_WRAPPER_H
#define _CUDA__STD__CUDA_API_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)
#  include <cuda_runtime_api.h>
#endif // !_CCCL_CUDA_COMPILER_NVCC && !_CCCL_CUDA_COMPILER_NVHPC

#include <cuda/std/__exception/cuda_error.h>

#define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)           \
  {                                                    \
    const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    switch (__status)                                  \
    {                                                  \
      case ::cudaSuccess:                              \
        break;                                         \
      default:                                         \
        ::cudaGetLastError();                          \
        ::cuda::__throw_cuda_error(__status, _MSG);    \
    }                                                  \
  }

#define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)        \
  {                                                    \
    const ::cudaError_t __status = _NAME(__VA_ARGS__); \
    _LIBCUDACXX_ASSERT(__status == cudaSuccess, _MSG); \
    (void) __status;                                   \
  }

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief `__cuda_set_device_wrapper` is a simple helper that sets the current device to a given target and resets it
//! back in its destructor
struct __cuda_set_device_wrapper
{
  int __target_device_   = 0;
  int __original_device_ = 0;

  __cuda_set_device_wrapper(const int __target_device)
      : __target_device_(__target_device)
  {
    _CCCL_TRY_CUDA_API(::cudaGetDevice, "Failed to query current device", &__original_device_);
    if (__original_device_ != __target_device_)
    {
      _CCCL_TRY_CUDA_API(::cudaSetDevice, "Failed to set device", __target_device_);
    }
  }

  ~__cuda_set_device_wrapper()
  {
    if (__original_device_ != __target_device_)
    {
      _CCCL_TRY_CUDA_API(::cudaSetDevice, "Failed to set device", __original_device_);
    }
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif //_CUDA__STD__CUDA_API_WRAPPER_H
