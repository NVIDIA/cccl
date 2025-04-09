//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_BASE_OF_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_BASE_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF)

template <class _Tp, class _Up>
inline constexpr bool is_pointer_interconvertible_base_of_v =
  _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF(_Tp, _Up);

template <class _Tp, class _Up>
struct is_pointer_interconvertible_base_of : bool_constant<is_pointer_interconvertible_base_of_v<_Tp, _Up>>
{};

#endif // _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_POINTER_INTERCONVERTIBLE_BASE_OF_H
