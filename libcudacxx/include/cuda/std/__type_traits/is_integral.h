//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_INTEGRAL) && !defined(_LIBCUDACXX_USE_IS_INTEGRAL_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_integral : public integral_constant<bool, _CCCL_BUILTIN_IS_INTEGRAL(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_integral_v = _CCCL_BUILTIN_IS_INTEGRAL(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else // ^^^ _CCCL_BUILTIN_IS_INTEGRAL ^^^ / vvv !_CCCL_BUILTIN_IS_INTEGRAL vvv

template <class _Tp>
struct __cccl_is_integral : public false_type
{};
template <>
struct __cccl_is_integral<bool> : public true_type
{};
template <>
struct __cccl_is_integral<char> : public true_type
{};
template <>
struct __cccl_is_integral<signed char> : public true_type
{};
template <>
struct __cccl_is_integral<unsigned char> : public true_type
{};
template <>
struct __cccl_is_integral<wchar_t> : public true_type
{};
#  ifndef _LIBCUDACXX_NO_HAS_CHAR8_T
template <>
struct __cccl_is_integral<char8_t> : public true_type
{};
#  endif
#  ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
template <>
struct __cccl_is_integral<char16_t> : public true_type
{};
template <>
struct __cccl_is_integral<char32_t> : public true_type
{};
#  endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
template <>
struct __cccl_is_integral<short> : public true_type
{};
template <>
struct __cccl_is_integral<unsigned short> : public true_type
{};
template <>
struct __cccl_is_integral<int> : public true_type
{};
template <>
struct __cccl_is_integral<unsigned int> : public true_type
{};
template <>
struct __cccl_is_integral<long> : public true_type
{};
template <>
struct __cccl_is_integral<unsigned long> : public true_type
{};
template <>
struct __cccl_is_integral<long long> : public true_type
{};
template <>
struct __cccl_is_integral<unsigned long long> : public true_type
{};
#  ifndef _LIBCUDACXX_HAS_NO_INT128
template <>
struct __cccl_is_integral<__int128_t> : public true_type
{};
template <>
struct __cccl_is_integral<__uint128_t> : public true_type
{};
#  endif

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_integral
    : public integral_constant<bool, __cccl_is_integral<remove_cv_t<_Tp>>::value>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_integral_v = is_integral<_Tp>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // !_CCCL_BUILTIN_IS_INTEGRAL

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_INTEGRAL_H
