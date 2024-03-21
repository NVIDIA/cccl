// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_COMPLEX_NVFP16_H
#define _LIBCUDACXX___CUDA_COMPLEX_NVFP16_H

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

#if defined(_LIBCUDACXX_HAS_NVFP16)

#  include <cuda_fp16.h>

#  include <cuda/std/detail/libcxx/include/__complex/traits.h>
#  include <cuda/std/detail/libcxx/include/__cuda/cmath_nvfp16.h>
#  include <cuda/std/detail/libcxx/include/__type_traits/integral_constant.h>
#  include <cuda/std/cmath>
#  include <cuda/std/complex>

#  if !defined(_CCCL_COMPILER_NVRTC)
#    include <sstream> // for std::basic_ostringstream
#  endif // !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <>
struct __is_nvfp16<__half> : true_type
{};

template <>
struct __complex_alignment<__half> : integral_constant<size_t, alignof(__half2)>
{};

template <>
struct __type_to_vector<__half>
{
  using __type = __half2;
};

template <>
struct __libcpp_complex_overload_traits<__half, false, false>
{
  typedef __half _ValueType;
  typedef complex<__half> _ComplexType;
};

// We can utilize vectorized operations for those operators
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half>&
operator+=(complex<__half>& __lhs, const complex<__half>& __rhs) noexcept
{
  reinterpret_cast<__half2&>(__lhs) += reinterpret_cast<const __half2&>(__rhs);
  return __lhs;
}

inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half>&
operator-=(complex<__half>& __lhs, const complex<__half>& __rhs) noexcept
{
  reinterpret_cast<__half2&>(__lhs) -= reinterpret_cast<const __half2&>(__rhs);
  return __lhs;
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half arg(__half __re)
{
  return _CUDA_VSTD::atan2f(__half(0), __re);
}

// We have performance issues with some trigonometric functions with __half
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> asinh(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::asinh(complex<float>{__x})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> acosh(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::acosh(complex<float>{__x})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> atanh(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::atanh(complex<float>{__x})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> acos(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::acos(complex<float>{__x})};
}

#  if !defined(_LIBCUDACXX_HAS_NO_LOCALIZATION) && !defined(_CCCL_COMPILER_NVRTC)
template <class _CharT, class _Traits>
::std::basic_istream<_CharT, _Traits>& operator>>(::std::basic_istream<_CharT, _Traits>& __is, complex<__half>& __x)
{
  ::std::complex<float> __temp;
  __is >> __temp;
  __x = __temp;
  return __is;
}

template <class _CharT, class _Traits>
::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const complex<__half>& __x)
{
  return __os << complex<float>{__x};
}
#  endif // !_LIBCUDACXX_HAS_NO_LOCALIZATION && !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVFP16

#endif // _LIBCUDACXX___CUDA_COMPLEX_NVFP16_H
