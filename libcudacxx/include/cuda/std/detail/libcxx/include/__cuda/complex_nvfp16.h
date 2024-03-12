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

#  include "../__cuda/cmath_nvfp16.h"
#  include "../__type_traits/integral_constant.h"
#  include "../__type_traits/enable_if.h"
#  include "../__type_traits/is_arithmetic.h"
#  include "../__type_traits/is_same.h"
#  include "../cmath"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <>
struct __is_nvfp16<__half> : true_type
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
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_COMPLEX_ALIGNAS(alignof(__half2)) complex<__half>
{
  __half2 __repr;

public:
  typedef __half value_type;

  _LIBCUDACXX_INLINE_VISIBILITY complex(__half __re = 0.0f, __half __im = 0.0f)
      : __repr(__re, __im)
  {}
  template <class _Int, typename = __enable_if_t<is_arithmetic<_Int>::value>>
  _LIBCUDACXX_INLINE_VISIBILITY explicit complex(_Int __re = _Int(), _Int __im = _Int())
      : __repr(__re, __im)
  {}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // narrowing conversions

  _LIBCUDACXX_INLINE_VISIBILITY explicit complex(const complex<float>& __c)
      : __repr(__c.real(), __c.imag())
  {}
  _LIBCUDACXX_INLINE_VISIBILITY explicit complex(const complex<double>& __c)
      : __repr(__c.real(), __c.imag())
  {}

_CCCL_DIAG_POP

#  if !defined(_CCCL_COMPILER_NVRTC)
  template <class _Up>
  _LIBCUDACXX_INLINE_VISIBILITY complex(const ::std::complex<_Up>& __other)
      : __repr(_LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other), _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other))
  {}

  template <class _Up>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator=(const ::std::complex<_Up>& __other)
  {
    __repr.x = _LIBCUDACXX_ACCESS_STD_COMPLEX_REAL(__other);
    __repr.y = _LIBCUDACXX_ACCESS_STD_COMPLEX_IMAG(__other);
    return *this;
  }
#  endif // !defined(_CCCL_COMPILER_NVRTC)

  _LIBCUDACXX_INLINE_VISIBILITY __half real() const
  {
    return __repr.x;
  }
  _LIBCUDACXX_INLINE_VISIBILITY __half imag() const
  {
    return __repr.y;
  }

  _LIBCUDACXX_INLINE_VISIBILITY void real(value_type __re)
  {
    __repr.x = __re;
  }
  _LIBCUDACXX_INLINE_VISIBILITY void imag(value_type __im)
  {
    __repr.y = __im;
  }

  _LIBCUDACXX_INLINE_VISIBILITY complex& operator=(__half __re)
  {
    __repr.x = __re;
    __repr.y = value_type();
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(__half __re)
  {
    __repr.x += __re;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(__half __re)
  {
    __repr.x -= __re;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(__half __re)
  {
    __repr.x *= __re;
    __repr.y *= __re;
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(__half __re)
  {
    __repr.x /= __re;
    __repr.y /= __re;
    return *this;
  }

  template <class _Xp>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator=(const complex<_Xp>& __c)
  {
    __repr.x = __c.real();
    __repr.y = __c.imag();
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const complex& __c)
  {
    __repr += __c.__repr;
    return *this;
  }
  template <class _Xp>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator+=(const complex<_Xp>& __c)
  {
    __repr.x += __c.real();
    __repr.y += __c.imag();
    return *this;
  }
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const complex& __c)
  {
    __repr -= __c.__repr;
    return *this;
  }
  template <class _Xp>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator-=(const complex<_Xp>& __c)
  {
    __repr.x -= __c.real();
    __repr.y -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator*=(const complex<_Xp>& __c)
  {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _LIBCUDACXX_INLINE_VISIBILITY complex& operator/=(const complex<_Xp>& __c)
  {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

inline _LIBCUDACXX_INLINE_VISIBILITY complex<float>::complex(const complex<__half>& __c)
    : __re_(__c.real())
    , __im_(__c.imag())
{}

inline _LIBCUDACXX_INLINE_VISIBILITY complex<double>::complex(const complex<__half>& __c)
    : __re_(__c.real())
    , __im_(__c.imag())
{}

inline _LIBCUDACXX_INLINE_VISIBILITY __half arg(__half __re)
{
  return _CUDA_VSTD::atan2f(__half(0), __re);
}

// We have performance issues with some trigonometric functions with __half
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> asinh(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::asinh(complex<float>{__x.real(), __x.imag()})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> acosh(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::acosh(complex<float>{__x.real(), __x.imag()})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> atanh(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::atanh(complex<float>{__x.real(), __x.imag()})};
}
template <>
inline _LIBCUDACXX_INLINE_VISIBILITY complex<__half> acos(const complex<__half>& __x)
{
  return complex<__half>{_CUDA_VSTD::acos(complex<float>{__x.real(), __x.imag()})};
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVFP16

#endif // _LIBCUDACXX___CUDA_COMPLEX_NVFP16_H
