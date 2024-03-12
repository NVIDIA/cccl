//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SIFT_DOWN_H
#define _LIBCUDACXX___ALGORITHM_SIFT_DOWN_H

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

#include "../__algorithm/iterator_operations.h"
#include "../__assert"
#include "../__iterator/iterator_traits.h"
#include "../__utility/move.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void __sift_down(
  _RandomAccessIterator __first,
  _Compare&& __comp,
  typename iterator_traits<_RandomAccessIterator>::difference_type __len,
  _RandomAccessIterator __start)
{
  using _Ops = _IterOps<_AlgPolicy>;

  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  typedef typename iterator_traits<_RandomAccessIterator>::value_type value_type;
  // left-child of __start is at 2 * __start + 1
  // right-child of __start is at 2 * __start + 2
  difference_type __child = __start - __first;

  if (__len < 2 || (__len - 2) / 2 < __child)
  {
    return;
  }

  __child                         = 2 * __child + 1;
  _RandomAccessIterator __child_i = __first + __child;

  if ((__child + 1) < __len && __comp(*__child_i, *(__child_i + difference_type(1))))
  {
    // right-child exists and is greater than left-child
    ++__child_i;
    ++__child;
  }

  // check if we are in heap-order
  if (__comp(*__child_i, *__start))
  {
    // we are, __start is larger than its largest child
    return;
  }

  value_type __top(_Ops::__iter_move(__start));
  do
  {
    // we are not in heap-order, swap the parent with its largest child
    *__start = _Ops::__iter_move(__child_i);
    __start  = __child_i;

    if ((__len - 2) / 2 < __child)
    {
      break;
    }

    // recompute the child based off of the updated parent
    __child   = 2 * __child + 1;
    __child_i = __first + __child;

    if ((__child + 1) < __len && __comp(*__child_i, *(__child_i + difference_type(1))))
    {
      // right-child exists and is greater than left-child
      ++__child_i;
      ++__child;
    }

    // check if we are in heap-order
  } while (!__comp(*__child_i, __top));
  *__start = _CUDA_VSTD::move(__top);
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _RandomAccessIterator __floyd_sift_down(
  _RandomAccessIterator __first,
  _Compare&& __comp,
  typename iterator_traits<_RandomAccessIterator>::difference_type __len)
{
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  _LIBCUDACXX_ASSERT(__len >= 2, "shouldn't be called unless __len >= 2");

  _RandomAccessIterator __hole    = __first;
  _RandomAccessIterator __child_i = __first;
  difference_type __child         = 0;

  while (true)
  {
    __child_i += difference_type(__child + 1);
    __child = 2 * __child + 1;

    if ((__child + 1) < __len && __comp(*__child_i, *(__child_i + difference_type(1))))
    {
      // right-child exists and is greater than left-child
      ++__child_i;
      ++__child;
    }

    // swap __hole with its largest child
    *__hole = _IterOps<_AlgPolicy>::__iter_move(__child_i);
    __hole  = __child_i;

    // if __hole is now a leaf, we're done
    if (__child > (__len - 2) / 2)
    {
      return __hole;
    }
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_SIFT_DOWN_H
