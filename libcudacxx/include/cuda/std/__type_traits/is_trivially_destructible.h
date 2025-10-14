//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_destructible.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(is_trivially_destructible) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE(...) __is_trivially_destructible(__VA_ARGS__)
#endif // has __is_trivially_destructible || msvc || nvrtc

#if _CCCL_CHECK_BUILTIN(has_trivial_destructor) || _CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR(...) __has_trivial_destructor(__VA_ARGS__)
#endif // has __has_trivial_destructor || gcc || msvc || nvrtc

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE)

template <class _Tp>
inline constexpr bool is_trivially_destructible_v = _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE ^^^ / vvv !_CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE vvv

template <class _Tp>
inline constexpr bool is_trivially_destructible_v = is_destructible_v<_Tp> && _CCCL_BUILTIN_HAS_TRIVIAL_DESTRUCTOR(_Tp);

#endif // ^^^ !_CCCL_BUILTIN_IS_TRIVIALLY_DESTRUCTIBLE ^^^

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_trivially_destructible : bool_constant<is_trivially_destructible_v<_Tp>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_TRIVIALLY_DESTRUCTIBLE_H
