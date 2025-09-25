//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H
#define _CUDA_STD___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H

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

#define _CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(...) __has_virtual_destructor(__VA_ARGS__)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT has_virtual_destructor : bool_constant<_CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(_Tp)>
{};

template <class _Tp>
inline constexpr bool has_virtual_destructor_v = _CCCL_BUILTIN_HAS_VIRTUAL_DESTRUCTOR(_Tp);

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_HAS_VIRTUAL_DESTRUCTOR_H
