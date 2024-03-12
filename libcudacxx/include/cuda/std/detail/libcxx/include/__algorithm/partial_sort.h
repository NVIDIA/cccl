//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PARTIAL_SORT_H
#define _LIBCUDACXX___ALGORITHM_PARTIAL_SORT_H

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

#include "../__algorithm/comp.h"
#include "../__algorithm/comp_ref_type.h"
#include "../__algorithm/iterator_operations.h"
#include "../__algorithm/make_heap.h"
#include "../__algorithm/sift_down.h"
#include "../__algorithm/sort_heap.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__utility/move.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Sentinel>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _RandomAccessIterator
__partial_sort_impl(_RandomAccessIterator __first, _RandomAccessIterator __middle, _Sentinel __last, _Compare&& __comp)
{
  if (__first == __middle)
  {
    return _IterOps<_AlgPolicy>::next(__middle, __last);
  }

  _CUDA_VSTD::__make_heap<_AlgPolicy>(__first, __middle, __comp);

  typename iterator_traits<_RandomAccessIterator>::difference_type __len = __middle - __first;
  _RandomAccessIterator __i                                              = __middle;
  for (; __i != __last; ++__i)
  {
    if (__comp(*__i, *__first))
    {
      _IterOps<_AlgPolicy>::iter_swap(__i, __first);
      _CUDA_VSTD::__sift_down<_AlgPolicy>(__first, __comp, __len, __first);
    }
  }
  _CUDA_VSTD::__sort_heap<_AlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__middle), __comp);

  return __i;
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Sentinel>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _RandomAccessIterator
__partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _Sentinel __last, _Compare& __comp)
{
  if (__first == __middle)
  {
    return _IterOps<_AlgPolicy>::next(__middle, __last);
  }

  return _CUDA_VSTD::__partial_sort_impl<_AlgPolicy>(
    __first, __middle, __last, static_cast<__comp_ref_type<_Compare> >(__comp));
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void partial_sort(
  _RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp)
{
  static_assert(_LIBCUDACXX_TRAIT(is_copy_constructible, _RandomAccessIterator),
                "Iterators must be copy constructible.");
  static_assert(_LIBCUDACXX_TRAIT(is_copy_assignable, _RandomAccessIterator), "Iterators must be copy assignable.");

  (void) _CUDA_VSTD::__partial_sort<_ClassicAlgPolicy>(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__middle), _CUDA_VSTD::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last)
{
  _CUDA_VSTD::partial_sort(__first, __middle, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_PARTIAL_SORT_H
