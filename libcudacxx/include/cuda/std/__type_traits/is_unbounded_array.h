//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_UNBOUNDED_ARRAY_H
#define _CUDA_STD___TYPE_TRAITS_IS_UNBOUNDED_ARRAY_H

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

#if _CCCL_CHECK_BUILTIN(is_unbounded_array)
#  define _CCCL_BUILTIN_IS_UNBOUNDED_ARRAY(...) __is_unbounded_array(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_unbounded_array)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_UNBOUNDED_ARRAY)

template <class _Tp>
inline constexpr bool is_unbounded_array_v = _CCCL_BUILTIN_IS_UNBOUNDED_ARRAY(_Tp);

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_unbounded_array : bool_constant<_CCCL_BUILTIN_IS_UNBOUNDED_ARRAY(_Tp)>
{};

#else // ^^^ _CCCL_BUILTIN_IS_UNBOUNDED_ARRAY ^^^ / vvv !_CCCL_BUILTIN_IS_UNBOUNDED_ARRAY vvv

template <class _Tp>
inline constexpr bool is_unbounded_array_v = false;

template <class _Tp>
inline constexpr bool is_unbounded_array_v<_Tp[]> = true;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_unbounded_array : bool_constant<is_unbounded_array_v<_Tp>>
{};

#endif // ^^^ !_CCCL_BUILTIN_IS_UNBOUNDED_ARRAY ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_UNBOUNDED_ARRAY_H
