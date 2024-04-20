// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H
#define _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H

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

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/detail/libcxx/include/stdexcept>

#include <nv/target>
#if !defined(_CCCL_COMPILER_NVRTC)
#  include <cstdio>
#endif // !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
class cuda_error : public _CUDA_VSTD_NOVERSION::runtime_error
{
public:
  cuda_error(::cudaError_t __status, const char* __msg) noexcept
      : _CUDA_VSTD_NOVERSION::runtime_error("")
  {
    ::snprintf(
      const_cast<char*>(this->what()), _CUDA_VSTD::__libcpp_refstring::__length, "cudaError %d: %s", __status, __msg);
  }
};

_CCCL_NORETURN inline _LIBCUDACXX_INLINE_VISIBILITY void __throw_cuda_error(::cudaError_t __status, const char* __msg)
{
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (throw ::cuda::cuda_error(__status, __msg);),
                    ((void) __status; (void) __msg; _CUDA_VSTD_NOVERSION::terminate();))
#else
  (void) __status;
  (void) __msg;
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_LIBCUDACXX_NO_EXCEPTIONS
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H
