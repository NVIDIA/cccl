//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_CMP_H
#define _LIBCUDACXX___UTILITY_CMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/limits>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class... _Up>
using __is_same_as_any = __fold_or<_CCCL_TRAIT(is_same, _Tp, _Up)...>;

template <class _Tp>
struct __is_safe_integral_cmp
    : bool_constant<_CCCL_TRAIT(is_integral, _Tp)
                    && !__is_same_as_any<_Tp,
                                         bool,
                                         char,
                                         char16_t,
                                         char32_t
#ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
                                         ,
                                         char8_t
#endif // _LIBCUDACXX_NO_HAS_CHAR8_T
#ifndef _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
                                         ,
                                         wchar_t
#endif // _LIBCUDACXX_HAS_NO_WIDE_CHARACTERS
                                         >::value>
{};

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool cmp_equal(_Tp __t, _Up __u) noexcept
{
#if _CCCL_STD_VER >= 2017
  _CCCL_IF_CONSTEXPR (_CCCL_TRAIT(is_signed, _Tp) == _CCCL_TRAIT(is_signed, _Up))
  {
    return __t == __u;
  }
  _CCCL_ELSE_IF_CONSTEXPR (_CCCL_TRAIT(is_signed, _Tp))
  {
    return __t < 0 ? false : make_unsigned_t<_Tp>(__t) == __u;
  }
  else
  {
    return __u < 0 ? false : __t == make_unsigned_t<_Up>(__u);
  }
#else // ^^^ C++17 ^^^ / vvv C++14 vvv
  return ((_CCCL_TRAIT(is_signed, _Tp) == _CCCL_TRAIT(is_signed, _Up))
            ? (__t == __u)
            : (_CCCL_TRAIT(is_signed, _Tp) ? (__t < 0 ? false : make_unsigned_t<_Tp>(__t) == __u)
                                           : (__u < 0 ? false : __t == make_unsigned_t<_Up>(__u))));
#endif // _CCCL_STD_VER <= 2014
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool cmp_not_equal(_Tp __t, _Up __u) noexcept
{
  return !_CUDA_VSTD::cmp_equal(__t, __u);
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool cmp_less(_Tp __t, _Up __u) noexcept
{
#if _CCCL_STD_VER >= 2017
  _CCCL_IF_CONSTEXPR (_CCCL_TRAIT(is_signed, _Tp) == _CCCL_TRAIT(is_signed, _Up))
  {
    return __t < __u;
  }
  _CCCL_ELSE_IF_CONSTEXPR (_CCCL_TRAIT(is_signed, _Tp))
  {
    return __t < 0 ? true : make_unsigned_t<_Tp>(__t) < __u;
  }
  else
  {
    return __u < 0 ? false : __t < make_unsigned_t<_Up>(__u);
  }
#else // ^^^ C++17 ^^^ / vvv C++14 vvv
  return ((_CCCL_TRAIT(is_signed, _Tp) == _CCCL_TRAIT(is_signed, _Up))
            ? (__t < __u)
            : (_CCCL_TRAIT(is_signed, _Tp) ? (__t < 0 ? true : make_unsigned_t<_Tp>(__t) < __u)
                                           : (__u < 0 ? false : __t < make_unsigned_t<_Up>(__u))));
#endif // _CCCL_STD_VER <= 2014
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool cmp_greater(_Tp __t, _Up __u) noexcept
{
  return _CUDA_VSTD::cmp_less(__u, __t);
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool cmp_less_equal(_Tp __t, _Up __u) noexcept
{
  return !_CUDA_VSTD::cmp_greater(__t, __u);
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool cmp_greater_equal(_Tp __t, _Up __u) noexcept
{
  return !_CUDA_VSTD::cmp_less(__t, __u);
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Up)
_LIBCUDACXX_REQUIRES(__is_safe_integral_cmp<_Tp>::value _LIBCUDACXX_AND __is_safe_integral_cmp<_Up>::value)
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool in_range(_Up __u) noexcept
{
  return _CUDA_VSTD::cmp_less_equal(__u, numeric_limits<_Tp>::max())
      && _CUDA_VSTD::cmp_greater_equal(__u, numeric_limits<_Tp>::min());
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___UTILITY_CMP_H
