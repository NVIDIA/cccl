//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_VOID_H
#define _CUDA_STD___TYPE_TRAITS_IS_VOID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(is_void)
#  define _CCCL_BUILTIN_IS_VOID(...) __is_void(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_void)

// clang + nvcc fails with '_Tp does not refer to a value' or 'identifier __is_void is undefined'
#if _CCCL_COMPILER(CLANG) && _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_IS_VOID
#endif // _CCCL_COMPILER(CLANG) && _CCCL_CUDA_COMPILER(NVCC)

// if we put this trait to cuda::std::, it colides with libstdc++, putting it to to global namespace seems to work fine
#if defined(_CCCL_BUILTIN_IS_VOID)
template <class _Tp>
inline constexpr bool __cccl_is_void_v = _CCCL_BUILTIN_IS_VOID(_Tp);
#endif // _CCCL_BUILTIN_IS_VOID

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_VOID)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_void : bool_constant<::__cccl_is_void_v<_Tp>>
{};

template <class _Tp>
inline constexpr bool is_void_v = ::__cccl_is_void_v<_Tp>;

#else // ^^^ _CCCL_BUILTIN_IS_VOID ^^^ / vvv !_CCCL_BUILTIN_IS_VOID vvv

template <class _Tp>
inline constexpr bool is_void_v = is_same_v<remove_cv_t<_Tp>, void>;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_void : public bool_constant<is_void_v<_Tp>>
{};

#endif // ^^^ !_CCCL_BUILTIN_IS_VOID ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_VOID_H
