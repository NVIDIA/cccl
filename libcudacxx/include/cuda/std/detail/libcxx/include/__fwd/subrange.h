//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_SUBRANGE_H
#define _LIBCUDACXX___FWD_SUBRANGE_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__iterator/concepts.h"

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

enum class _LIBCUDACXX_ENUM_VIS subrange_kind : bool
{
  unsized,
  sized
};

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI
#  if _CCCL_STD_VER >= 2020
template <input_or_output_iterator _Iter,
          sentinel_for<_Iter> _Sent = _Iter,
          subrange_kind _Kind       = sized_sentinel_for<_Sent, _Iter> ? subrange_kind::sized : subrange_kind::unsized>
  requires(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>)
class _LIBCUDACXX_TEMPLATE_VIS subrange;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter,
          class _Sent         = _Iter,
          subrange_kind _Kind = sized_sentinel_for<_Sent, _Iter> ? subrange_kind::sized : subrange_kind::unsized,
          enable_if_t<input_or_output_iterator<_Iter>, int>                                      = 0,
          enable_if_t<sentinel_for<_Sent, _Iter>, int>                                           = 0,
          enable_if_t<(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>), int> = 0>
class _LIBCUDACXX_TEMPLATE_VIS subrange;
#  endif // _CCCL_STD_VER <= 2017

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___FWD_SUBRANGE_H
