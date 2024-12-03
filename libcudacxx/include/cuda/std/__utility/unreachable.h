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
#if defined(__CUDA_ARCH__) && !defined(_CCCL_CUDA_COMPILER_CLANG)
#  if _CCCL_CUDACC_BELOW(11, 2)
  __trap();
#  elif _CCCL_CUDACC_BELOW(11, 3)
  __builtin_assume(false);
#  else
  __builtin_unreachable();
#  endif // CUDACC above 11.4
#elif _CCCL_COMPILER(MSVC)
  __assume(0);
#else
  __builtin_unreachable();
#endif
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif
