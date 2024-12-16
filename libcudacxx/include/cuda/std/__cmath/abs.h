// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ABS_H
#define _LIBCUDACXX___CMATH_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using ::fabs;
using ::fabsf;
#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
using ::fabsl;
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float abs(float __val) noexcept
{
  return _CUDA_VSTD::fabsf(__val);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double abs(double __val) noexcept
{
  return _CUDA_VSTD::fabs(__val);
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double abs(long double __val) noexcept
{
  return _CUDA_VSTD::fabsl(__val);
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_ABS_H
