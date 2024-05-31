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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_NVFP16)

#  include <cuda_fp16.h>

#  include <cuda/std/__complex/vector_support.h>
#  include <cuda/std/__cuda/cmath_nvfp16.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_constructible.h>
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

template <>
class _LIBCUDACXX_TEMPLATE_VIS _CCCL_ALIGNAS(alignof(__half2)) complex<__half>
{
  __half2 __repr_;

  template <class _Up>
  friend class complex;

public:
  using value_type = __half;

  _LIBCUDACXX_INLINE_VISIBILITY complex(const value_type& __re = value_type(), const value_type& __im = value_type())
      : __repr_(__re, __im)
  {}

  template <class _Up, __enable_if_t<__complex_can_implicitly_construct<value_type, _Up>::value, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY complex(const complex<_Up>& __c)
      : __repr_(static_cast<value_type>(__c.real()), static_cast<value_type>(__c.imag()))
  {}

  template <class _Up,
            __enable_if_t<!__complex_can_implicitly_construct<value_type, _Up>::value, int> = 0,
            __enable_if_t<_CCCL_TRAIT(is_constructible, value_type, _Up), int>              = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit complex(const complex<_Up>& __c)
      : __repr_(static_cast<value_type>(__c.real()), static_cast<value_type>(__c.imag()))
  {}

  _LIBCUDACXX_INLINE_VISIBILITY complex& operator=(const value_type& __re)
  {
    __repr_.x = __re;
    __repr_.y = value_type();
    return *this;
  }

  template <class _Up>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator=(const complex<_Up>& __c)
  {
    __repr_.x = __c.real();
    __repr_.y = __c.imag();
    return *this;
  }

#  if !defined(_CCCL_COMPILER_NVRTC)
  template <class _Up>
  _LIBCUDACXX_INLINE_VISIBILITY complex(const ::std::complex<_Up>& __other)
      : __repr_(_LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other), _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other))
  {}

  template <class _Up>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator=(const ::std::complex<_Up>& __other)
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

  _LIBCUDACXX_INLINE_VISIBILITY value_type real() const
  {
    return __repr_.x;
  }
  _LIBCUDACXX_INLINE_VISIBILITY value_type imag() const
  {
    return __repr_.y;
  }

  _LIBCUDACXX_INLINE_VISIBILITY void real(value_type __re)
  {
    __repr_.x = __re;
  }
  _LIBCUDACXX_INLINE_VISIBILITY void imag(value_type __im)
  {
    __repr_.y = __im;
  }

  // Those additional volatile overloads are meant to help with reductions in thrust
  _LIBCUDACXX_INLINE_VISIBILITY value_type real() const volatile
  {
    return __repr_.x;
  }
  _LIBCUDACXX_INLINE_VISIBILITY value_type imag() const volatile
  {
    return __repr_.y;
  }

  _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const value_type& __re)
  {
    __repr_.x += __re;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const value_type& __re)
  {
    __repr_.x -= __re;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const value_type& __re)
  {
    __repr_.x *= __re;
    __repr_.y *= __re;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const value_type& __re)
  {
    __repr_.x /= __re;
    __repr_.y /= __re;
    return *this;
  }

  // We can utilize vectorized operations for those operators
  _LIBCUDACXX_INLINE_VISIBILITY friend complex& operator+=(complex& __lhs, const complex& __rhs) noexcept
  {
    __lhs.__repr_ = __hadd2(__lhs.__repr_, __rhs.__repr_);
    return __lhs;
  }

  _LIBCUDACXX_INLINE_VISIBILITY friend complex& operator-=(complex& __lhs, const complex& __rhs) noexcept
  {
    __lhs.__repr_ = __hsub2(__lhs.__repr_, __rhs.__repr_);
    return __lhs;
  }

  _LIBCUDACXX_INLINE_VISIBILITY friend bool operator==(const complex& __lhs, const complex& __rhs) noexcept
  {
    return __hbeq2(__lhs.__repr_, __rhs.__repr_);
  }
};

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
