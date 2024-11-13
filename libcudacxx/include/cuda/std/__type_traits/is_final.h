//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_FINAL)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_final : public integral_constant<bool, _CCCL_BUILTIN_IS_FINAL(_Tp)>
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_final_v = _CCCL_BUILTIN_IS_FINAL(_Tp);
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#else // ^^^ _CCCL_BUILTIN_IS_FINAL ^^^ / vvv !_CCCL_BUILTIN_IS_FINAL vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_final : public false_type
{};

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool is_final_v = false;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // !_CCCL_BUILTIN_IS_FINAL

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H
