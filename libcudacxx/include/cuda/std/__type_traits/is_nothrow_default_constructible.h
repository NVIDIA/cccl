//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DEFAULT_CONSTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DEFAULT_CONSTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_nothrow_constructible.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_default_constructible
    : public integral_constant<bool, _LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE(_Tp)>
{};

#  if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class... _Args>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_default_constructible_v = _LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE(_Tp);
#  endif

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_default_constructible : public is_nothrow_constructible<_Tp>
{};

#  if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_default_constructible_v = is_nothrow_constructible<_Tp>::value;
#  endif

#endif // defined(_LIBCUDACXX_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NOTHROW_DEFAULT_CONSTRUCTIBLE_H
