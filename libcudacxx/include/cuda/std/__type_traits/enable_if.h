//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-25 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
#define _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool, class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT enable_if
{};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT enable_if<true, _Tp>
{
  using type = _Tp;
};

#if _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(GCC, <, 8)

template <bool _Bp, class _Tp = void>
using enable_if_t _CCCL_NODEBUG_ALIAS = typename enable_if<_Bp, _Tp>::type;

#else // ^^^ standard implementation ^^^ / vvv optimized implementation

// Optimized enable_if_t implementation that does not instantiate a type every time
template <bool>
struct __enable_if_t_impl
{};

template <>
struct __enable_if_t_impl<true>
{
  template <class _Tp>
  using type = _Tp;
};

template <bool _Bp, class _Tp = void>
using enable_if_t _CCCL_NODEBUG_ALIAS = typename __enable_if_t_impl<_Bp>::template type<_Tp>;

#endif // !_CCCL_COMPILER(NVHPC) && !_CCCL_COMPILER(GCC, <, 8)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
