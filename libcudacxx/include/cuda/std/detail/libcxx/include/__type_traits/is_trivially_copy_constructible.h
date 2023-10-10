//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_COPY_CONSTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_COPY_CONSTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/add_const.h"
#include "../__type_traits/add_lvalue_reference.h"
#include "../__type_traits/is_trivially_constructible.h"

_CCCL_IMPLICIT_SYSTEM_HEADER

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_TRIVIALLY_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_CONSTRUCTIBLE_FALLBACK)

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_trivially_copy_constructible
    : public integral_constant<bool,
        _LIBCUDACXX_IS_TRIVIALLY_CONSTRUCTIBLE(_Tp, __add_lvalue_reference_t<typename add_const<_Tp>::type>)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_trivially_copy_constructible_v =
    _LIBCUDACXX_IS_TRIVIALLY_CONSTRUCTIBLE(_Tp, __add_lvalue_reference_t<typename add_const<_Tp>::type>);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_trivially_copy_constructible
    : public is_trivially_constructible<_Tp, __add_lvalue_reference_t<typename add_const<_Tp>::type>>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_trivially_copy_constructible_v = is_trivially_copy_constructible<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_TRIVIALLY_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_CONSTRUCTIBLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_COPY_CONSTRUCTIBLE_H
