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

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__exception/cuda_error.h>

#if _CCCL_HAS_CUDA_COMPILER
#  define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)           \
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

#  define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)        \
    {                                                    \
      const ::cudaError_t __status = _NAME(__VA_ARGS__); \
      _CCCL_ASSERT(__status == cudaSuccess, _MSG);       \
      (void) __status;                                   \
    }
#else // ^^^ _CCCL_HAS_CUDA_COMPILER ^^^ / vvv !_CCCL_HAS_CUDA_COMPILER vvv
#  define _CCCL_TRY_CUDA_API(_NAME, _MSG, ...)
#  define _CCCL_ASSERT_CUDA_API(_NAME, _MSG, ...)
#endif // !_CCCL_HAS_CUDA_COMPILER

#endif //_CUDA__STD__CUDA_API_WRAPPER_H
