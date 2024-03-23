// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_NVBF16_H
#define _LIBCUDACXX___COMPLEX_NVBF16_H

#ifndef __cuda_std__
#  include <config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_NVBF16)

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP

#  include <cuda/std/cmath>
#  include <cuda/std/complex>
#  include <cuda/std/detail/libcxx/include/__complex/vector_support.h>
#  include <cuda/std/detail/libcxx/include/__cuda/cmath_nvbf16.h>
#  include <cuda/std/detail/libcxx/include/__type_traits/integral_constant.h>

#  if !defined(_CCCL_COMPILER_NVRTC)
#    include <sstream> // for std::basic_ostringstream
#  endif // !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <>
struct __is_nvbf16<__nv_bfloat16> : true_type
{};

template <>
struct __complex_alignment<__nv_bfloat16> : integral_constant<size_t, alignof(__nv_bfloat162)>
{};

template <>
struct __type_to_vector<__nv_bfloat16>
{
  using __type = __nv_bfloat162;
};

template <>
struct __libcpp_complex_overload_traits<__nv_bfloat16, false, false>
{
  typedef __nv_bfloat16 _ValueType;
  typedef complex<__nv_bfloat16> _ComplexType;
};

// We can utilize vectorized operations for those operators
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__nv_bfloat16>&
operator+=(complex<__nv_bfloat16>& __lhs, const complex<__nv_bfloat16>& __rhs) noexcept
{
  reinterpret_cast<__nv_bfloat162&>(__lhs) += reinterpret_cast<const __nv_bfloat162&>(__rhs);
  return __lhs;
}

inline _LIBCUDACXX_INLINE_VISIBILITY complex<__nv_bfloat16>&
operator-=(complex<__nv_bfloat16>& __lhs, const complex<__nv_bfloat16>& __rhs) noexcept
{
  reinterpret_cast<__nv_bfloat162&>(__lhs) -= reinterpret_cast<const __nv_bfloat162&>(__rhs);
  return __lhs;
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 arg(__nv_bfloat16 __re)
{
  return _CUDA_VSTD::atan2f(__nv_bfloat16(0), __re);
}

// We have performance issues with some trigonometric functions with __nv_bfloat16
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__nv_bfloat16> asinh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::asinh(complex<float>{__x})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__nv_bfloat16> acosh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::acosh(complex<float>{__x})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__nv_bfloat16> atanh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::atanh(complex<float>{__x})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__nv_bfloat16> acos(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::acos(complex<float>{__x})};
}

#  if !defined(_CCCL_COMPILER_NVRTC)
template <class _CharT, class _Traits>
::std::basic_istream<_CharT, _Traits>&
operator>>(::std::basic_istream<_CharT, _Traits>& __is, complex<__nv_bfloat16>& __x)
{
  ::std::complex<float> __temp;
  __is >> __temp;
  __x = __temp;
  return __is;
}

template <class _CharT, class _Traits>
::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const complex<__nv_bfloat16>& __x)
{
  return __os << complex<float>{__x};
}
#  endif // !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVBF16

#endif // _LIBCUDACXX___COMPLEX_NVBF16_H
