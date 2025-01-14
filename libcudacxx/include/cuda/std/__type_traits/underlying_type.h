//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H
#define _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_enum.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_UNDERLYING_TYPE) && !defined(_LIBCUDACXX_USE_UNDERLYING_TYPE_FALLBACK)

template <class _Tp, bool = is_enum<_Tp>::value>
struct __underlying_type_impl;

template <class _Tp>
struct __underlying_type_impl<_Tp, false>
{};

template <class _Tp>
struct __underlying_type_impl<_Tp, true>
{
  typedef _CCCL_BUILTIN_UNDERLYING_TYPE(_Tp) type;
};

template <class _Tp>
struct underlying_type : __underlying_type_impl<_Tp, is_enum<_Tp>::value>
{};

template <class _Tp>
using underlying_type_t _CCCL_NODEBUG_ALIAS = typename underlying_type<_Tp>::type;

#else // ^^^ _CCCL_BUILTIN_UNDERLYING_TYPE ^^^ / vvv !_CCCL_BUILTIN_UNDERLYING_TYPE vvv

template <class _Tp, bool _Support = false>
struct underlying_type
{
  static_assert(_Support,
                "The underyling_type trait requires compiler "
                "support. Either no such support exists or "
                "libcu++ does not know how to use it.");
};

#endif // !_CCCL_BUILTIN_UNDERLYING_TYPE

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H
