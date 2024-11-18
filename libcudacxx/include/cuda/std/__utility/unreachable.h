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

_CCCL_NORETURN _CCCL_HIDE_FROM_ABI _CCCL_HOST void __unreachable_host_impl()
{
#if defined(_CCCL_COMPILER_MSVC)
  __assume(0);
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
  __builtin_unreachable();
#endif // !_CCCL_COMPILER_MSVC
}

#if defined(_CCCL_CUDA_COMPILER)
_CCCL_NORETURN _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __unreachable_device_impl()
{
#  if defined(_CCCL_CUDA_COMPILER_NVCC)
#    if _CCCL_CUDACC_BELOW(11, 2)
  __trap();
#    elif _CCCL_CUDACC_BELOW(11, 3)
  __builtin_assume(false);
#    else
  __builtin_unreachable();
#    endif
#  else // ^^^ _CCCL_CUDA_COMPILER_NVCC ^^^ / vvv !_CCCL_CUDA_COMPILER_NVCC vvv
  __builtin_unreachable();
#  endif // !_CCCL_CUDA_COMPILER_NVCC
}
#endif // _CCCL_CUDA_COMPILER

_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void unreachable()
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, _CUDA_VSTD::__unreachable_host_impl();, _CUDA_VSTD::__unreachable_device_impl();)
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif
