//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SHIFT_LEFT_H
#define _LIBCUDACXX___ALGORITHM_SHIFT_LEFT_H

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

#include "../__algorithm/move.h"
#include "../__iterator/iterator_traits.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ForwardIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator __shift_left(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  random_access_iterator_tag)
{
  _ForwardIterator __m = __first;
  if (__n >= __last - __first)
  {
    return __first;
  }
  __m += __n;
  return _CUDA_VSTD::move(__m, __last, __first);
}

template <class _ForwardIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator __shift_left(
  _ForwardIterator __first,
  _ForwardIterator __last,
  typename iterator_traits<_ForwardIterator>::difference_type __n,
  forward_iterator_tag)
{
  _ForwardIterator __m = __first;
  for (; __n > 0; --__n)
  {
    if (__m == __last)
    {
      return __first;
    }
    ++__m;
  }
  return _CUDA_VSTD::move(__m, __last, __first);
}

template <class _ForwardIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator shift_left(
  _ForwardIterator __first, _ForwardIterator __last, typename iterator_traits<_ForwardIterator>::difference_type __n)
{
  if (__n == 0)
  {
    return __last;
  }

  using _IterCategory = typename iterator_traits<_ForwardIterator>::iterator_category;
  return __shift_left(__first, __last, __n, _IterCategory());
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_SHIFT_LEFT_H
