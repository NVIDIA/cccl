// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_DIV_H
#define _LIBCUDACXX___CSTDLIB_DIV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT div_t
{
  int quot;
  int rem;
};

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr div_t div(int __x, int __y) noexcept
{
  return {__x / __y, __x % __y};
}

struct _CCCL_TYPE_VISIBILITY_DEFAULT ldiv_t
{
  long quot;
  long rem;
};

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ldiv_t ldiv(long __x, long __y) noexcept
{
  return {__x / __y, __x % __y};
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ldiv_t div(long __x, long __y) noexcept
{
  return _CUDA_VSTD::ldiv(__x, __y);
}

struct _CCCL_TYPE_VISIBILITY_DEFAULT lldiv_t
{
  long long quot;
  long long rem;
};

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr lldiv_t lldiv(long long __x, long long __y) noexcept
{
  return {__x / __y, __x % __y};
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr lldiv_t div(long long __x, long long __y) noexcept
{
  return _CUDA_VSTD::lldiv(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_DIV_H
