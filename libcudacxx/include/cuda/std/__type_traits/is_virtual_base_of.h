//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_VIRTUAL_BASE_OF_H
#define _CUDA_STD___TYPE_TRAITS_IS_VIRTUAL_BASE_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF)

template <class _Base, class _Derived>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_virtual_base_of : bool_constant<_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF(_Base, _Derived)>
{};

template <class _Base, class _Derived>
inline constexpr bool is_virtual_base_of_v = _CCCL_BUILTIN_IS_VIRTUAL_BASE_OF(_Base, _Derived);

#else // ^^^ _CCCL_BUILTIN_IS_VIRTUAL_BASE_OF ^^^ / vvv !_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF vvv

template <class _Base, class _Derived>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_virtual_base_of : false_type
{
  static_assert(__always_false_v<_Base>,
                "cuda::std::is_virtual_base_of: compiler support is required to implement this trait");
};

template <class _Base, class _Derived>
inline constexpr bool is_virtual_base_of_v = is_virtual_base_of<_Base, _Derived>::value;

#endif // ^^^ !_CCCL_BUILTIN_IS_VIRTUAL_BASE_OF ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_VIRTUAL_BASE_OF_H
