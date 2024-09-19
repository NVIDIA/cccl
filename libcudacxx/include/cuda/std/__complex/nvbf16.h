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

#include <cuda/std/detail/__config>

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

#  include <cuda/std/__complex/vector_support.h>
#  include <cuda/std/__cuda/cmath_nvbf16.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_constructible.h>
#  include <cuda/std/__type_traits/is_extended_floating_point.h>
#  include <cuda/std/cmath>
#  include <cuda/std/complex>

#  if !defined(_CCCL_COMPILER_NVRTC)
#    include <sstream> // for std::basic_ostringstream
#  endif // !_CCCL_COMPILER_NVRTC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

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

// This is a workaround against the user defining macros __CUDA_NO_BFLOAT16_CONVERSIONS__ __CUDA_NO_BFLOAT16_OPERATORS__
template <>
struct __complex_can_implicitly_construct<__nv_bfloat16, float> : true_type
{};

template <>
struct __complex_can_implicitly_construct<__nv_bfloat16, double> : true_type
{};

template <>
struct __complex_can_implicitly_construct<float, __nv_bfloat16> : true_type
{};

template <>
struct __complex_can_implicitly_construct<double, __nv_bfloat16> : true_type
{};

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __convert_to_bfloat16(const _Tp& __value) noexcept
{
  return __value;
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __convert_to_bfloat16(const float& __value) noexcept
{
  return __float2bfloat16(__value);
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __convert_to_bfloat16(const double& __value) noexcept
{
  return __double2bfloat16(__value);
}

template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_ALIGNAS(alignof(__nv_bfloat162)) complex<__nv_bfloat16>
{
  __nv_bfloat162 __repr_;

  template <class _Up>
  friend class complex;

public:
  using value_type = __nv_bfloat16;

  _LIBCUDACXX_HIDE_FROM_ABI complex(const value_type& __re = value_type(), const value_type& __im = value_type())
      : __repr_(__re, __im)
  {}

  template <class _Up, __enable_if_t<__complex_can_implicitly_construct<value_type, _Up>::value, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI complex(const complex<_Up>& __c)
      : __repr_(__convert_to_bfloat16(__c.real()), __convert_to_bfloat16(__c.imag()))
  {}

  template <class _Up,
            __enable_if_t<!__complex_can_implicitly_construct<value_type, _Up>::value, int> = 0,
            __enable_if_t<_CCCL_TRAIT(is_constructible, value_type, _Up), int>              = 0>
  _LIBCUDACXX_HIDE_FROM_ABI explicit complex(const complex<_Up>& __c)
      : __repr_(__convert_to_bfloat16(__c.real()), __convert_to_bfloat16(__c.imag()))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI complex& operator=(const value_type& __re)
  {
    __repr_.x = __re;
    __repr_.y = value_type();
    return *this;
  }

  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI complex& operator=(const complex<_Up>& __c)
  {
    __repr_.x = __convert_to_bfloat16(__c.real());
    __repr_.y = __convert_to_bfloat16(__c.imag());
    return *this;
  }

#  if !defined(_CCCL_COMPILER_NVRTC)
  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI complex(const ::std::complex<_Up>& __other)
      : __repr_(_LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other), _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other))
  {}

  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI complex& operator=(const ::std::complex<_Up>& __other)
  {
    __repr_.x = _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other);
    __repr_.y = _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other);
    return *this;
  }

  _CCCL_HOST operator ::std::complex<value_type>() const
  {
    return {__repr_.x, __repr_.y};
  }
#  endif // !_CCCL_COMPILER_NVRTC

  _LIBCUDACXX_HIDE_FROM_ABI value_type real() const
  {
    return __repr_.x;
  }
  _LIBCUDACXX_HIDE_FROM_ABI value_type imag() const
  {
    return __repr_.y;
  }

  _LIBCUDACXX_HIDE_FROM_ABI void real(value_type __re)
  {
    __repr_.x = __re;
  }
  _LIBCUDACXX_HIDE_FROM_ABI void imag(value_type __im)
  {
    __repr_.y = __im;
  }

  // Those additional volatile overloads are meant to help with reductions in thrust
  _LIBCUDACXX_HIDE_FROM_ABI value_type real() const volatile
  {
    return __repr_.x;
  }
  _LIBCUDACXX_HIDE_FROM_ABI value_type imag() const volatile
  {
    return __repr_.y;
  }

  _LIBCUDACXX_HIDE_FROM_ABI complex& operator+=(const value_type& __re)
  {
    __repr_.x = __hadd(__repr_.x, __re);
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI complex& operator-=(const value_type& __re)
  {
    __repr_.x = __hsub(__repr_.x, __re);
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI complex& operator*=(const value_type& __re)
  {
    __repr_.x = __hmul(__repr_.x, __re);
    __repr_.y = __hmul(__repr_.y, __re);
    return *this;
  }
  _LIBCUDACXX_HIDE_FROM_ABI complex& operator/=(const value_type& __re)
  {
    __repr_.x = __hdiv(__repr_.x, __re);
    __repr_.y = __hdiv(__repr_.y, __re);
    return *this;
  }

  // We can utilize vectorized operations for those operators
  _LIBCUDACXX_HIDE_FROM_ABI friend complex& operator+=(complex& __lhs, const complex& __rhs) noexcept
  {
    __lhs.__repr_ = __hadd2(__lhs.__repr_, __rhs.__repr_);
    return __lhs;
  }

  _LIBCUDACXX_HIDE_FROM_ABI friend complex& operator-=(complex& __lhs, const complex& __rhs) noexcept
  {
    __lhs.__repr_ = __hsub2(__lhs.__repr_, __rhs.__repr_);
    return __lhs;
  }

  _LIBCUDACXX_HIDE_FROM_ABI friend bool operator==(const complex& __lhs, const complex& __rhs) noexcept
  {
    return __hbeq2(__lhs.__repr_, __rhs.__repr_);
  }
};

template <> // complex<float>
template <> // complex<__half>
_LIBCUDACXX_HIDE_FROM_ABI complex<float>::complex(const complex<__nv_bfloat16>& __c)
    : __re_(__bfloat162float(__c.real()))
    , __im_(__bfloat162float(__c.imag()))
{}

template <> // complex<double>
template <> // complex<__half>
_LIBCUDACXX_HIDE_FROM_ABI complex<double>::complex(const complex<__nv_bfloat16>& __c)
    : __re_(__bfloat162float(__c.real()))
    , __im_(__bfloat162float(__c.imag()))
{}

template <> // complex<float>
template <> // complex<__nv_bfloat16>
_LIBCUDACXX_HIDE_FROM_ABI complex<float>& complex<float>::operator=(const complex<__nv_bfloat16>& __c)
{
  __re_ = __bfloat162float(__c.real());
  __im_ = __bfloat162float(__c.imag());
  return *this;
}

template <> // complex<double>
template <> // complex<__nv_bfloat16>
_LIBCUDACXX_HIDE_FROM_ABI complex<double>& complex<double>::operator=(const complex<__nv_bfloat16>& __c)
{
  __re_ = __bfloat162float(__c.real());
  __im_ = __bfloat162float(__c.imag());
  return *this;
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 arg(__nv_bfloat16 __re)
{
  return _CUDA_VSTD::atan2(__int2bfloat16_rn(0), __re);
}

// We have performance issues with some trigonometric functions with __nv_bfloat16
template <>
_LIBCUDACXX_HIDE_FROM_ABI complex<__nv_bfloat16> asinh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::asinh(complex<float>{__x})};
}
template <>
_LIBCUDACXX_HIDE_FROM_ABI complex<__nv_bfloat16> acosh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::acosh(complex<float>{__x})};
}
template <>
_LIBCUDACXX_HIDE_FROM_ABI complex<__nv_bfloat16> atanh(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{_CUDA_VSTD::atanh(complex<float>{__x})};
}
template <>
_LIBCUDACXX_HIDE_FROM_ABI complex<__nv_bfloat16> acos(const complex<__nv_bfloat16>& __x)
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
