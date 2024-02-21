//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MOVE_BACKWARD_H
#define _LIBCUDACXX___ALGORITHM_MOVE_BACKWARD_H

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
#include "../__algorithm/unwrap_iter.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_trivially_move_assignable.h"
#include "../__type_traits/remove_const.h"
#include "../__utility/pair.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _BidirectionalIterator, class _OutputIterator>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_BidirectionalIterator, _OutputIterator>
  __move_backward(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __result)
{
  while (__first != __last)
  {
    *--__result = _IterOps<_AlgPolicy>::__iter_move(--__last);
  }
  return {__first, __result};
}

template <class _AlgPolicy,
          class _Tp,
          class _Up,
          __enable_if_t<_LIBCUDACXX_TRAIT(is_same, __remove_const_t<_Tp>, _Up), int> = 0,
          __enable_if_t<_LIBCUDACXX_TRAIT(is_trivially_move_assignable, _Up), int>   = 0>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair<_Tp*, _Up*>
__move_backward(_Tp* __first, _Tp* __last, _Up* __result)
{
  const ptrdiff_t __n = __last - __first;
  if (__n > 0)
  {
    if (__dispatch_memmove(__result - __n, __first, __n))
    {
      return {__last, __result - __n};
    }
    for (ptrdiff_t __i = 1; __i <= __n; ++__i)
    {
      *(__result - __i) = _IterOps<_AlgPolicy>::__iter_move((__last - __i));
    }
  }
  return {__last, __result - __n};
}

template <class _BidirectionalIterator1, class _BidirectionalIterator2>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _BidirectionalIterator2
move_backward(_BidirectionalIterator1 __first, _BidirectionalIterator1 __last, _BidirectionalIterator2 __result)
{
  return _CUDA_VSTD::__move_backward<_ClassicAlgPolicy>(
           __unwrap_iter(__first), __unwrap_iter(__last), __unwrap_iter(__result))
    .second;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_MOVE_BACKWARD_H
