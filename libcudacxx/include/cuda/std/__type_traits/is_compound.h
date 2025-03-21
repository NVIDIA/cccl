//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_COMPOUND_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_COMPOUND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_fundamental.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_COMPOUND) && !defined(_LIBCUDACXX_USE_IS_COMPOUND_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_compound : public integral_constant<bool, _CCCL_BUILTIN_IS_COMPOUND(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
inline constexpr bool is_compound_v = _CCCL_BUILTIN_IS_COMPOUND(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else // ^^^ _CCCL_BUILTIN_IS_COMPOUND ^^^ / vvv !_CCCL_BUILTIN_IS_COMPOUND vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_compound : public integral_constant<bool, !is_fundamental<_Tp>::value>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
inline constexpr bool is_compound_v = is_compound<_Tp>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // !_CCCL_BUILTIN_IS_COMPOUND

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_COMPOUND_H
