// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_DANGLING_H
#define _LIBCUDACXX___RANGES_DANGLING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/enable_if.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

struct dangling
{
  _CCCL_HIDE_FROM_ABI dangling() = default;
  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr dangling(_Args&&...) noexcept
  {}
};

#  if _CCCL_STD_VER >= 2020
template <range _Rp>
using borrowed_iterator_t = _If<borrowed_range<_Rp>, iterator_t<_Rp>, dangling>;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Rp>
using borrowed_iterator_t = enable_if_t<range<_Rp>, _If<borrowed_range<_Rp>, iterator_t<_Rp>, dangling>>;
#  endif // _CCCL_STD_VER <= 2017

// borrowed_subrange_t defined in <__ranges/subrange.h>

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX___RANGES_DANGLING_H
