//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_COPY_ASSIGNABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_COPY_ASSIGNABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/add_const.h"
#include "../__type_traits/add_lvalue_reference.h"
#include "../__type_traits/is_assignable.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_copy_assignable
    : public is_assignable<__add_lvalue_reference_t<_Tp>,
                           __add_lvalue_reference_t<typename add_const<_Tp>::type>> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_copy_assignable_v = is_copy_assignable<_Tp>::value;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_COPY_ASSIGNABLE_H
