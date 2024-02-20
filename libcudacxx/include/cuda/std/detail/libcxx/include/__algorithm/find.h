//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FIND_H
#define _LIBCUDACXX___ALGORITHM_FIND_H

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

#include "../__functional/invoke.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// generic implementation
template <class _Iter, class _Sent, class _Tp, class _Proj>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  _Iter
  __find_impl(_Iter __first, _Sent __last, const _Tp& __value, _Proj& __proj)
{
  for (; __first != __last; ++__first)
  {
    if (_CUDA_VSTD::__invoke(__proj, *__first) == __value)
    {
      break;
    }
  }
  return __first;
}

template <class _InputIterator, class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  _InputIterator
  find(_InputIterator __first, _InputIterator __last, const _Tp& __value_)
{
  for (; __first != __last; ++__first)
  {
    if (*__first == __value_)
    {
      break;
    }
  }
  return __first;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_FIND_H
