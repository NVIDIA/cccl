//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SCOPED_ENUM_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SCOPED_ENUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/underlying_type.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, bool = _CCCL_TRAIT(is_enum, _Tp)>
struct __is_scoped_enum_helper : false_type
{};

template <class _Tp>
struct __is_scoped_enum_helper<_Tp, true>
    : public bool_constant<!_CCCL_TRAIT(is_convertible, _Tp, underlying_type_t<_Tp>)>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scoped_enum : public __is_scoped_enum_helper<_Tp>
{};

#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_scoped_enum_v = is_scoped_enum<_Tp>::value;
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SCOPED_ENUM_H
