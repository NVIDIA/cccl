//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_NULL_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_NULL_POINTER_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_nullptr_t_impl : public false_type
{};
template <>
struct __is_nullptr_t_impl<nullptr_t> : public true_type
{};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS __is_nullptr_t : public __is_nullptr_t_impl<__remove_cv_t<_Tp>>
{};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_null_pointer : public __is_nullptr_t_impl<__remove_cv_t<_Tp>>
{};

#if _CCCL_STD_VER > 2011 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_null_pointer_v = is_null_pointer<_Tp>::value;
#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NULL_POINTER_H
