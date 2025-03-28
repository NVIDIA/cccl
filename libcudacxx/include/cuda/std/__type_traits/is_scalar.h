//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_member_pointer.h>
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_SCALAR) && !defined(_LIBCUDACXX_USE_IS_SCALAR_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scalar : public integral_constant<bool, _CCCL_BUILTIN_IS_SCALAR(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_scalar_v = _CCCL_BUILTIN_IS_SCALAR(_Tp);

#else

template <class _Tp>
struct __is_block : false_type
{};
#  if defined(_LIBCUDACXX_HAS_EXTENSION_BLOCKS)
template <class _Rp, class... _Args>
struct __is_block<_Rp (^)(_Args...)> : true_type
{};
#  endif

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scalar
    : public integral_constant<bool,
                               is_arithmetic<_Tp>::value || is_member_pointer<_Tp>::value || is_pointer<_Tp>::value
                                 || __is_nullptr_t<_Tp>::value || __is_block<_Tp>::value || is_enum<_Tp>::value>
{};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scalar<nullptr_t> : public true_type
{};

template <class _Tp>
inline constexpr bool is_scalar_v = is_scalar<_Tp>::value;

#endif // defined(_CCCL_BUILTIN_IS_SCALAR) && !defined(_LIBCUDACXX_USE_IS_SCALAR_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SCALAR_H
