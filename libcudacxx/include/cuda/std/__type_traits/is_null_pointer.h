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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

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
struct _CCCL_TYPE_VISIBILITY_DEFAULT __is_nullptr_t : public __is_nullptr_t_impl<remove_cv_t<_Tp>>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_null_pointer : public __is_nullptr_t_impl<remove_cv_t<_Tp>>
{};

#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_null_pointer_v = is_null_pointer<_Tp>::value;
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_NULL_POINTER_H
