//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ILOG
#define _LIBCUDACXX___CMATH_ILOG

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/integral.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/make_unsigned.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int ilog2(_Tp __t) noexcept
{
  static_assert(_CUDA_VSTD::__cccl_is_integer_v<_Tp>,
                "ilog2() argument type must be an integer type");
  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
  _CCCL_ASSERT(__t > 0, "ilog2() argument must be strictly positive");
  auto __ret = _CUDA_VSTD::__bit_log2(static_cast<_Up>(__t));
  _CCCL_ASSUME(__ret <= _CUDA_VSTD::numeric_limits<_Tp>::digits);
  return __ret;
}

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int ilog10(_Tp __t) noexcept
{
  static_assert(_CUDA_VSTD::is_integral_v<_Tp> && !_CUDA_VSTD::is_same_v<_Tp, bool>,
                "ilog2() argument type must be an integer type");
  _CCCL_ASSERT(__t > 0, "ilog10() argument must be strictly positive");
  constexpr auto __reciprocal_log2_10 = 1.0f / 3.321928094f; // 1 / log2(10)
  auto __log2                         = _CUDA_VSTD::__bit_log2(__t);
  auto __ret                          = static_cast<int>(__log2 * __reciprocal_log2_10);
  _CCCL_ASSUME(__ret <= _CUDA_VSTD::numeric_limits<_Tp>::digits / 3);
  return __ret;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CMATH_ILOG
