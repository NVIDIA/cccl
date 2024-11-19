//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_LITERAL_TYPE
#define _LIBCUDACXX___TYPE_TRAITS_IS_LITERAL_TYPE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/remove_all_extents.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_LITERAL) && !defined(_LIBCUDACXX_USE_IS_LITERAL_FALLBACK)
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
_LIBCUDACXX_DEPRECATED_IN_CXX17 is_literal_type : public integral_constant<bool, _CCCL_BUILTIN_IS_LITERAL(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_DEPRECATED_IN_CXX17 _CCCL_INLINE_VAR constexpr bool is_literal_type_v = __is_literal_type(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _LIBCUDACXX_DEPRECATED_IN_CXX17 is_literal_type
    : public integral_constant<bool,
                               is_scalar<remove_all_extents_t<_Tp>>::value
                                 || is_reference<remove_all_extents_t<_Tp>>::value>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_DEPRECATED_IN_CXX17 _CCCL_INLINE_VAR constexpr bool is_literal_type_v = is_literal_type<_Tp>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // defined(_CCCL_BUILTIN_IS_LITERAL) && !defined(_LIBCUDACXX_USE_IS_LITERAL_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_LITERAL_TYPE
