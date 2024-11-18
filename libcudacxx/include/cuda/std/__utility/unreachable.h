//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_UNREACHABLE_H
#define _LIBCUDACXX___UTILITY_UNREACHABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void unreachable()
{
#if defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_HOST_UNREACHABLE_IMPL() __assume(0)
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#  define _CCCL_HOST_UNREACHABLE_IMPL() __builtin_unreachable()
#endif // !_CCCL_COMPILER_MSVC

#if defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_DEVICE_UNREACHABLE_IMPL() __builtin_unreachable()
#elif defined(__CUDA_ARCH__)
#  if _CCCL_CUDACC_BELOW(11, 2)
#    define _CCCL_DEVICE_UNREACHABLE_IMPL() __trap()
#  elif _CCCL_CUDACC_BELOW(11, 3)
#    define _CCCL_DEVICE_UNREACHABLE_IMPL() __builtin_assume(false)
#  else
#    define _CCCL_DEVICE_UNREACHABLE_IMPL() __builtin_unreachable()
#  endif // CUDACC above 11.4
#else // ^^^ __CUDA_ARCH__ ^^^ / vvv !__CUDA_ARCH__ vvv
#  define _CCCL_DEVICE_UNREACHABLE_IMPL() __builtin_unreachable()
#endif // !__CUDA_ARCH__

  NV_IF_ELSE_TARGET(NV_IS_HOST, _CCCL_HOST_UNREACHABLE_IMPL();, _CCCL_DEVICE_UNREACHABLE_IMPL();)

#undef _CCCL_HOST_UNREACHABLE_IMPL
#undef _CCCL_DEVICE_UNREACHABLE_IMPL
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif
