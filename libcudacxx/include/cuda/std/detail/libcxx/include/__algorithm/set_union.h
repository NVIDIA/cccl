//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SET_UNION_H
#define _LIBCUDACXX___ALGORITHM_SET_UNION_H

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
#include "../__algorithm/copy.h"
#include "../__algorithm/iterator_operations.h"
#include "../__iterator/iterator_traits.h"
#include "../__utility/move.h"
#include "../__utility/pair.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InIter1, class _InIter2, class _OutIter>
struct __set_union_result {
  _InIter1 __in1_;
  _InIter2 __in2_;
  _OutIter __out_;

  // need a constructor as C++03 aggregate init is hard
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  __set_union_result(_InIter1&& __in_iter1, _InIter2&& __in_iter2, _OutIter&& __out_iter)
      : __in1_(_CUDA_VSTD::move(__in_iter1)), __in2_(_CUDA_VSTD::move(__in_iter2)), __out_(_CUDA_VSTD::move(__out_iter)) {}
};

template <class _AlgPolicy, class _Compare, class _InIter1, class _Sent1, class _InIter2, class _Sent2, class _OutIter>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __set_union_result<_InIter1, _InIter2, _OutIter> __set_union(
    _InIter1 __first1, _Sent1 __last1, _InIter2 __first2, _Sent2 __last2, _OutIter __result, _Compare&& __comp) {
  for (; __first1 != __last1; ++__result) {
    if (__first2 == __last2) {
      auto __ret1 = _CUDA_VSTD::__copy<_AlgPolicy>(_CUDA_VSTD::move(__first1), _CUDA_VSTD::move(__last1), _CUDA_VSTD::move(__result));
      return __set_union_result<_InIter1, _InIter2, _OutIter>(
          _CUDA_VSTD::move(__ret1.first), _CUDA_VSTD::move(__first2), _CUDA_VSTD::move((__ret1.second)));
    }
    if (__comp(*__first2, *__first1)) {
      *__result = *__first2;
      ++__first2;
    } else {
      if (!__comp(*__first1, *__first2)) {
        ++__first2;
      }
      *__result = *__first1;
      ++__first1;
    }
  }
  auto __ret2 = _CUDA_VSTD::__copy<_AlgPolicy>(_CUDA_VSTD::move(__first2), _CUDA_VSTD::move(__last2), _CUDA_VSTD::move(__result));
  return __set_union_result<_InIter1, _InIter2, _OutIter>(
      _CUDA_VSTD::move(__first1), _CUDA_VSTD::move(__ret2.first), _CUDA_VSTD::move((__ret2.second)));
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _OutputIterator set_union(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp) {
  return _CUDA_VSTD::__set_union<_ClassicAlgPolicy, __comp_ref_type<_Compare> >(
             _CUDA_VSTD::move(__first1),
             _CUDA_VSTD::move(__last1),
             _CUDA_VSTD::move(__first2),
             _CUDA_VSTD::move(__last2),
             _CUDA_VSTD::move(__result),
             __comp)
      .__out_;
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _OutputIterator set_union(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result) {
  return _CUDA_VSTD::set_union(
      _CUDA_VSTD::move(__first1),
      _CUDA_VSTD::move(__last1),
      _CUDA_VSTD::move(__first2),
      _CUDA_VSTD::move(__last2),
      _CUDA_VSTD::move(__result),
      __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_SET_UNION_H
