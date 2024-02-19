// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_DISTANCE_H
#define _LIBCUDACXX___ITERATOR_DISTANCE_H

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

#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/size.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/remove_cvref.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  typename iterator_traits<_InputIter>::difference_type
  __distance(_InputIter __first, _InputIter __last, input_iterator_tag)
{
  typename iterator_traits<_InputIter>::difference_type __r(0);
  for (; __first != __last; ++__first)
  {
    ++__r;
  }
  return __r;
}

template <class _RandIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  typename iterator_traits<_RandIter>::difference_type
  __distance(_RandIter __first, _RandIter __last, random_access_iterator_tag)
{
  return __last - __first;
}

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  typename iterator_traits<_InputIter>::difference_type
  distance(_InputIter __first, _InputIter __last)
{
  return _CUDA_VSTD::__distance(__first, __last, typename iterator_traits<_InputIter>::iterator_category());
}

_LIBCUDACXX_END_NAMESPACE_STD
#  if _CCCL_STD_VER > 2014 && !defined(_CCCL_COMPILER_MSVC_2017)

// [range.iter.op.distance]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__distance)
struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp)
  _LIBCUDACXX_REQUIRES((sentinel_for<_Sp, _Ip> && !sized_sentinel_for<_Sp, _Ip>) )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr iter_difference_t<_Ip>
  operator()(_Ip __first, _Sp __last) const
  {
    iter_difference_t<_Ip> __n = 0;
    while (__first != __last)
    {
      ++__first;
      ++__n;
    }
    return __n;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp)
  _LIBCUDACXX_REQUIRES((sized_sentinel_for<_Sp, decay_t<_Ip>>) )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr iter_difference_t<_Ip>
  operator()(_Ip&& __first, _Sp __last) const
  {
    if constexpr (sized_sentinel_for<_Sp, remove_cvref_t<_Ip>>)
    {
      return __last - __first;
    }
    else
    {
      return __last - decay_t<_Ip>(__first);
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _Rp)
  _LIBCUDACXX_REQUIRES((range<_Rp>) )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr range_difference_t<_Rp> operator()(_Rp&& __r) const
  {
    if constexpr (sized_range<_Rp>)
    {
      return static_cast<range_difference_t<_Rp>>(_CUDA_VRANGES::size(__r));
    }
    else
    {
      return operator()(_CUDA_VRANGES::begin(__r), _CUDA_VRANGES::end(__r));
    }
    _LIBCUDACXX_UNREACHABLE();
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto distance = __distance::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

#  endif // _CCCL_STD_VER > 2014  && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___ITERATOR_DISTANCE_H
