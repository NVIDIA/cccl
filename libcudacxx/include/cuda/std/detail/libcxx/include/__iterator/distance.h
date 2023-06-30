// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_DISTANCE_H
#define _LIBCUDACXX___ITERATOR_DISTANCE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#ifdef _LIBCUDACXX_HAS_RANGES
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/size.h"
#endif // _LIBCUDACXX_HAS_RANGES
#include "../__type_traits/decay.h"
#include "../__type_traits/remove_cvref.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
typename iterator_traits<_InputIter>::difference_type
__distance(_InputIter __first, _InputIter __last, input_iterator_tag)
{
    typename iterator_traits<_InputIter>::difference_type __r(0);
    for (; __first != __last; ++__first)
        ++__r;
    return __r;
}

template <class _RandIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
typename iterator_traits<_RandIter>::difference_type
__distance(_RandIter __first, _RandIter __last, random_access_iterator_tag)
{
    return __last - __first;
}

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
typename iterator_traits<_InputIter>::difference_type
distance(_InputIter __first, _InputIter __last)
{
    return _CUDA_VSTD::__distance(__first, __last, typename iterator_traits<_InputIter>::iterator_category());
}

_LIBCUDACXX_END_NAMESPACE_STD
#ifdef _LIBCUDACXX_HAS_RANGES
#if _LIBCUDACXX_STD_VER > 14

// [range.iter.op.distance]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__distance)
struct __fn {
  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp)
    (requires (sentinel_for<_Sp, _Ip> && !sized_sentinel_for<_Sp, _Ip>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr iter_difference_t<_Ip> operator()(_Ip __first, _Sp __last) const {
    iter_difference_t<_Ip> __n = 0;
    while (__first != __last) {
      ++__first;
      ++__n;
    }
    return __n;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp)
    (requires (sized_sentinel_for<_Sp, decay_t<_Ip>>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr iter_difference_t<_Ip> operator()(_Ip&& __first, _Sp __last) const {
    if constexpr (sized_sentinel_for<_Sp, remove_cvref_t<_Ip>>) {
      return __last - __first;
    } else {
      return __last - decay_t<_Ip>(__first);
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _Rp)
    (requires (range<_Rp>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr range_difference_t<_Rp> operator()(_Rp&& __r) const {
    if constexpr (sized_range<_Rp>) {
      return static_cast<range_difference_t<_Rp>>(_CUDA_VRANGES::size(__r));
    } else {
      return operator()(_CUDA_VRANGES::begin(__r), _CUDA_VRANGES::end(__r));
    }
    _LIBCUDACXX_UNREACHABLE();
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto distance = __distance::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX_STD_VER > 14
#endif // _LIBCUDACXX_HAS_RANGES

#endif // _LIBCUDACXX___ITERATOR_DISTANCE_H
