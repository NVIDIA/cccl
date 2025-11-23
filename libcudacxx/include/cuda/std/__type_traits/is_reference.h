//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_REFERENCE_H
#define _CUDA_STD___TYPE_TRAITS_IS_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_LVALUE_REFERENCE) && !defined(_LIBCUDACXX_USE_IS_LVALUE_REFERENCE_FALLBACK)  \
  && defined(_CCCL_BUILTIN_IS_RVALUE_REFERENCE) && !defined(_LIBCUDACXX_USE_IS_RVALUE_REFERENCE_FALLBACK) \
  && defined(_CCCL_BUILTIN_IS_REFERENCE) && !defined(_LIBCUDACXX_USE_IS_REFERENCE_FALLBACK)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_lvalue_reference : bool_constant<_CCCL_BUILTIN_IS_LVALUE_REFERENCE(_Tp)>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_rvalue_reference : bool_constant<_CCCL_BUILTIN_IS_RVALUE_REFERENCE(_Tp)>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_reference : bool_constant<_CCCL_BUILTIN_IS_REFERENCE(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_lvalue_reference_v = _CCCL_BUILTIN_IS_LVALUE_REFERENCE(_Tp);
template <class _Tp>
inline constexpr bool is_rvalue_reference_v = _CCCL_BUILTIN_IS_RVALUE_REFERENCE(_Tp);
template <class _Tp>
inline constexpr bool is_reference_v = _CCCL_BUILTIN_IS_REFERENCE(_Tp);

#else

template <class _Tp>
inline constexpr bool is_lvalue_reference_v = false;
template <class _Tp>
inline constexpr bool is_lvalue_reference_v<_Tp&> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_lvalue_reference : bool_constant<is_lvalue_reference_v<_Tp>>
{};

template <class _Tp>
inline constexpr bool is_rvalue_reference_v = false;
template <class _Tp>
inline constexpr bool is_rvalue_reference_v<_Tp&&> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_rvalue_reference : bool_constant<is_rvalue_reference_v<_Tp>>
{};

template <class _Tp>
inline constexpr bool is_reference_v = false;
template <class _Tp>
inline constexpr bool is_reference_v<_Tp&> = true;
template <class _Tp>
inline constexpr bool is_reference_v<_Tp&&> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_reference : bool_constant<is_reference_v<_Tp>>
{};

#endif // !_CCCL_BUILTIN_IS_LVALUE_REFERENCE

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_REFERENCE_H
