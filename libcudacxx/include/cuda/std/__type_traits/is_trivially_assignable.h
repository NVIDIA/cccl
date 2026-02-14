//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_TRIVIALLY_ASSIGNABLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_TRIVIALLY_ASSIGNABLE_H

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

#define _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(...) __is_trivially_assignable(__VA_ARGS__)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, class _Arg>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_trivially_assignable : bool_constant<_CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(_Tp, _Arg)>
{};

template <class _Tp, class _Arg>
inline constexpr bool is_trivially_assignable_v = _CCCL_BUILTIN_IS_TRIVIALLY_ASSIGNABLE(_Tp, _Arg);

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_TRIVIALLY_ASSIGNABLE_H
