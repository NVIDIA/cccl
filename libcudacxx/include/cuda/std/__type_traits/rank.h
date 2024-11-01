//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_RANK_H
#define _LIBCUDACXX___TYPE_TRAITS_RANK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_ARRAY_RANK) && !defined(_LIBCUDACXX_USE_ARRAY_RANK_FALLBACK) && 0

template <class _Tp>
struct rank : integral_constant<size_t, _CCCL_BUILTIN_ARRAY_RANK(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr size_t rank_v = _CCCL_BUILTIN_ARRAY_RANK(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT rank : public integral_constant<size_t, 0>
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT rank<_Tp[]> : public integral_constant<size_t, rank<_Tp>::value + 1>
{};
template <class _Tp, size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT rank<_Tp[_Np]> : public integral_constant<size_t, rank<_Tp>::value + 1>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr size_t rank_v = rank<_Tp>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // defined(_CCCL_BUILTIN_ARRAY_RANK) && !defined(_LIBCUDACXX_USE_ARRAY_RANK_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_RANK_H
