// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_ABS_H
#define _LIBCUDACXX___CSTDLIB_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__type_traits/is_constant_evaluated.h>

#if (_CCCL_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)) || _CCCL_HAS_CONSTEXPR_BUILTIN_ABS
#  define _CCCL_HAS_CONSTEXPR_INT_ABS 1
#  define _CCCL_CONSTEXPR_INT_ABS     constexpr
#else
#  define _CCCL_CONSTEXPR_INT_ABS
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_INT_ABS int abs(int __val) noexcept
{
#if !_CCCL_HAS_CONSTEXPR_BUILTIN_ABS
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    return (__val < 0) ? -__val : __val;
  }
#endif // !_CCCL_HAS_CONSTEXPR_BUILTIN_ABS

#if defined(_CCCL_BUILTIN_ABS)
  return _CCCL_BUILTIN_ABS(__val);
#else // ^^^ _CCCL_BUILTIN_ABS ^^^ / vvv !_CCCL_BUILTIN_ABS vvv
  return ::abs(__val);
#endif // !_CCCL_BUILTIN_ABS
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_INT_ABS long labs(long __val) noexcept
{
#if !_CCCL_HAS_CONSTEXPR_BUILTIN_ABS
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    return (__val < 0) ? -__val : __val;
  }
#endif // !_CCCL_HAS_CONSTEXPR_BUILTIN_ABS

#if defined(_CCCL_BUILTIN_ABS)
  return _CCCL_BUILTIN_LABS(__val);
#else // ^^^ _CCCL_BUILTIN_ABS ^^^ / vvv !_CCCL_BUILTIN_ABS vvv
  return ::labs(__val);
#endif // !_CCCL_BUILTIN_ABS
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_INT_ABS long long llabs(long long __val) noexcept
{
#if !_CCCL_HAS_CONSTEXPR_BUILTIN_ABS
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    return (__val < 0) ? -__val : __val;
  }
#endif // !_CCCL_HAS_CONSTEXPR_BUILTIN_ABS

#if defined(_CCCL_BUILTIN_ABS)
  return _CCCL_BUILTIN_LLABS(__val);
#else // ^^^ _CCCL_BUILTIN_ABS ^^^ / vvv !_CCCL_BUILTIN_ABS vvv
  return ::llabs(__val);
#endif // !_CCCL_BUILTIN_ABS
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_ABS_H
