//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_UNREACHABLE_H
#define __CCCL_UNREACHABLE_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/visibility.h>

#if _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_UNREACHABLE() __builtin_unreachable()
#elif defined(__CUDA_ARCH__)
#  if _CCCL_CUDACC_BELOW(11, 2)
#    define _CCCL_UNREACHABLE() __trap()
#  elif _CCCL_CUDACC_BELOW(11, 3)
#    define _CCCL_UNREACHABLE() __builtin_assume(false)
#  else
#    define _CCCL_UNREACHABLE() __builtin_unreachable()
#  endif // CUDACC above 11.4
#else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#  if _CCCL_COMPILER(MSVC2017)
template <class = void>
_LIBCUDACXX_HIDE_FROM_ABI __declspec(noreturn) void __cccl_unreachable_fallback()
{
  __assume(0);
}
#    define _CCCL_UNREACHABLE() __cccl_unreachable_fallback()
#  elif _CCCL_COMPILER(MSVC)
#    define _CCCL_UNREACHABLE() __assume(0)
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    define _CCCL_UNREACHABLE() __builtin_unreachable()
#  endif // !_CCCL_COMPILER(MSVC)
#endif // !__CUDA_ARCH__

#endif // __CCCL_UNREACHABLE_H
