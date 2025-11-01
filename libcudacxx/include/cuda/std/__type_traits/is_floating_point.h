//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_FLOATING_POINT_H
#define _CUDA_STD___TYPE_TRAITS_IS_FLOATING_POINT_H

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

#if _CCCL_HAS_BUILTIN(__is_floating_point)
#  define _CCCL_BUILTIN_IS_FLOATING_POINT(...) __is_floating_point(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_floating_point)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_FLOATING_POINT)
template <class _Tp>
inline constexpr bool is_floating_point_v = _CCCL_BUILTIN_IS_FLOATING_POINT(_Tp);
#else // ^^^ _CCCL_BUILTIN_IS_FLOATING_POINT ^^^ / vvv !_CCCL_BUILTIN_IS_FLOATING_POINT vvv
template <class _Tp>
inline constexpr bool is_floating_point_v = false;
template <class _Tp>
inline constexpr bool is_floating_point_v<const _Tp> = is_floating_point_v<_Tp>;
template <class _Tp>
inline constexpr bool is_floating_point_v<volatile _Tp> = is_floating_point_v<_Tp>;
template <class _Tp>
inline constexpr bool is_floating_point_v<const volatile _Tp> = is_floating_point_v<_Tp>;
template <>
inline constexpr bool is_floating_point_v<float> = true;
template <>
inline constexpr bool is_floating_point_v<double> = true;
template <>
inline constexpr bool is_floating_point_v<long double> = true;
#endif // ^^^ !_CCCL_BUILTIN_IS_FLOATING_POINT ^^^

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_floating_point : bool_constant<is_floating_point_v<_Tp>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_FLOATING_POINT_H
