//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PARTIAL_SORT_COPY_H
#define _LIBCUDACXX___ALGORITHM_PARTIAL_SORT_COPY_H

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
#include "../__algorithm/make_projected.h"
#include "../__algorithm/sift_down.h"
#include "../__algorithm/sort_heap.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/is_callable.h"
#include "../__utility/move.h"
#include "../__utility/pair.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy,
          class _Compare,
          class _InputIterator,
          class _Sentinel1,
          class _RandomAccessIterator,
          class _Sentinel2,
          class _Proj1,
          class _Proj2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair<_InputIterator, _RandomAccessIterator>
__partial_sort_copy(
  _InputIterator __first,
  _Sentinel1 __last,
  _RandomAccessIterator __result_first,
  _Sentinel2 __result_last,
  _Compare&& __comp,
  _Proj1&& __proj1,
  _Proj2&& __proj2)
{
  _RandomAccessIterator __r = __result_first;
  auto&& __projected_comp   = _CUDA_VSTD::__make_projected(__comp, __proj2);

  if (__r != __result_last)
  {
    for (; __first != __last && __r != __result_last; ++__first, (void) ++__r)
    {
      *__r = *__first;
    }
    _CUDA_VSTD::__make_heap<_AlgPolicy>(__result_first, __r, __projected_comp);
    typename iterator_traits<_RandomAccessIterator>::difference_type __len = __r - __result_first;
    for (; __first != __last; ++__first)
    {
      if (_CUDA_VSTD::__invoke(
            __comp, _CUDA_VSTD::__invoke(__proj1, *__first), _CUDA_VSTD::__invoke(__proj2, *__result_first)))
      {
        *__result_first = *__first;
        _CUDA_VSTD::__sift_down<_AlgPolicy>(__result_first, __projected_comp, __len, __result_first);
      }
    }
    _CUDA_VSTD::__sort_heap<_AlgPolicy>(__result_first, __r, __projected_comp);
  }

  return pair<_InputIterator, _RandomAccessIterator>(
    _IterOps<_AlgPolicy>::next(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last)), _CUDA_VSTD::move(__r));
}

template <class _InputIterator, class _RandomAccessIterator, class _Compare>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _RandomAccessIterator partial_sort_copy(
  _InputIterator __first,
  _InputIterator __last,
  _RandomAccessIterator __result_first,
  _RandomAccessIterator __result_last,
  _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first), decltype(*__result_first)>::value,
                "Comparator has to be callable");

  auto __result = _CUDA_VSTD::__partial_sort_copy<_ClassicAlgPolicy>(
    __first,
    __last,
    __result_first,
    __result_last,
    static_cast<__comp_ref_type<_Compare> >(__comp),
    __identity(),
    __identity());
  return __result.second;
}

template <class _InputIterator, class _RandomAccessIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _RandomAccessIterator partial_sort_copy(
  _InputIterator __first,
  _InputIterator __last,
  _RandomAccessIterator __result_first,
  _RandomAccessIterator __result_last)
{
  return _CUDA_VSTD::partial_sort_copy(__first, __last, __result_first, __result_last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_PARTIAL_SORT_COPY_H
