//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_REVERSE_H
#define _LIBCUDACXX___ALGORITHM_REVERSE_H

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

#include "../__algorithm/iter_swap.h"
#include "../__algorithm/iterator_operations.h"
#include "../__iterator/iterator_traits.h"
#include "../__utility/move.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _BidirectionalIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
__reverse_impl(_BidirectionalIterator __first, _BidirectionalIterator __last, bidirectional_iterator_tag)
{
  while (__first != __last)
  {
    if (__first == --__last)
    {
      break;
    }
    _IterOps<_AlgPolicy>::iter_swap(__first, __last);
    ++__first;
  }
}

template <class _AlgPolicy, class _RandomAccessIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
__reverse_impl(_RandomAccessIterator __first, _RandomAccessIterator __last, random_access_iterator_tag)
{
  if (__first != __last)
  {
    for (; __first < --__last; ++__first)
    {
      _IterOps<_AlgPolicy>::iter_swap(__first, __last);
    }
  }
}

template <class _AlgPolicy, class _BidirectionalIterator, class _Sentinel>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
__reverse(_BidirectionalIterator __first, _Sentinel __last)
{
  using _IterCategory = typename _IterOps<_AlgPolicy>::template __iterator_category<_BidirectionalIterator>;
  _CUDA_VSTD::__reverse_impl<_AlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), _IterCategory());
}

template <class _BidirectionalIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
reverse(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  _CUDA_VSTD::__reverse<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_REVERSE_H
