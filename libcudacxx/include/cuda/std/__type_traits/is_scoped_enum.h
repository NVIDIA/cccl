//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_SCOPED_ENUM_H
#define _CUDA_STD___TYPE_TRAITS_IS_SCOPED_ENUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/underlying_type.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(is_scoped_enum)
#  define _CCCL_BUILTIN_IS_SCOPED_ENUM(...) __is_scoped_enum(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_scoped_enum)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_SCOPED_ENUM)

template <class _Tp>
inline constexpr bool is_scoped_enum_v = _CCCL_BUILTIN_IS_SCOPED_ENUM(_Tp);

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scoped_enum : bool_constant<_CCCL_BUILTIN_IS_SCOPED_ENUM(_Tp)>
{};

#else // ^^^ _CCCL_BUILTIN_IS_SCOPED_ENUM ^^^ / vvv !_CCCL_BUILTIN_IS_SCOPED_ENUM vvv

template <class _Tp, bool = is_enum_v<_Tp>>
inline constexpr bool __cccl_is_scoped_enum_helper_v = false;

template <class _Tp>
inline constexpr bool __cccl_is_scoped_enum_helper_v<_Tp, true> = !is_convertible_v<_Tp, underlying_type_t<_Tp>>;

template <class _Tp>
inline constexpr bool is_scoped_enum_v = __cccl_is_scoped_enum_helper_v<_Tp>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_scoped_enum : bool_constant<__cccl_is_scoped_enum_helper_v<_Tp>>
{};

#endif // ^^^ !_CCCL_BUILTIN_IS_SCOPED_ENUM ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_SCOPED_ENUM_H
