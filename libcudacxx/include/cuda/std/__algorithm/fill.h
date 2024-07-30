//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FILL_H
#define _LIBCUDACXX___ALGORITHM_FILL_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__iterator/iterator_traits.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Tp>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 void
__fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value_, forward_iterator_tag)
{
  for (; __first != __last; ++__first)
  {
    *__first = __value_;
  }
}

template <class _RandomAccessIterator, class _Tp>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value_, random_access_iterator_tag)
{
  _CUDA_VSTD::fill_n(__first, __last - __first, __value_);
}

template <class _ForwardIterator, class _Tp>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value_)
{
  _CUDA_VSTD::__fill(__first, __last, __value_, typename iterator_traits<_ForwardIterator>::iterator_category());
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_FILL_H
