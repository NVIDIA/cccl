//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_FUNCTIONAL_H
#define _CUDA_STD___TYPE_TRAITS_IS_FUNCTIONAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_reference.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(is_function)
#  define _CCCL_BUILTIN_IS_FUNCTION(...) __is_function(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(is_function)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_FUNCTION)

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_function : bool_constant<_CCCL_BUILTIN_IS_FUNCTION(_Tp)>
{};

template <class _Tp>
inline constexpr bool is_function_v = _CCCL_BUILTIN_IS_FUNCTION(_Tp);

#else // ^^^ _CCCL_BUILTIN_IS_FUNCTION ^^^ / vvv !_CCCL_BUILTIN_IS_FUNCTION vvv

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4180) // qualifier applied to function type has no meaning; ignored

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_function : bool_constant<!(is_reference_v<_Tp> || is_const_v<const _Tp>)>
{};

template <class _Tp>
inline constexpr bool is_function_v = !(is_reference_v<_Tp> || is_const_v<const _Tp>);

_CCCL_DIAG_POP

#endif // ^^^ !_CCCL_BUILTIN_IS_FUNCTION ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_FUNCTIONAL_H
