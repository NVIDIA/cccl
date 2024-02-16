// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_PREV_H
#define _LIBCUDACXX___ITERATOR_PREV_H

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

#include "../__assert"
#include "../__iterator/advance.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/enable_if.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  __enable_if_t< __is_cpp17_input_iterator<_InputIter>::value, _InputIter >
  prev(_InputIter __x, typename iterator_traits<_InputIter>::difference_type __n = 1)
{
  _LIBCUDACXX_ASSERT(__n <= 0 || __is_cpp17_bidirectional_iterator<_InputIter>::value,
                     "Attempt to prev(it, +n) on a non-bidi iterator");
  _CUDA_VSTD::advance(__x, -__n);
  return __x;
}

_LIBCUDACXX_END_NAMESPACE_STD

#if _CCCL_STD_VER > 2014 && !defined(_CCCL_COMPILER_MSVC_2017)

// [range.iter.op.prev]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__prev)
struct __fn
{
  _LIBCUDACXX_TEMPLATE(class _Ip)
  _LIBCUDACXX_REQUIRES(bidirectional_iterator<_Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Ip operator()(_Ip __x) const
  {
    --__x;
    return __x;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip)
  _LIBCUDACXX_REQUIRES(bidirectional_iterator<_Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Ip
  operator()(_Ip __x, iter_difference_t<_Ip> __n) const
  {
    _CUDA_VRANGES::advance(__x, -__n);
    return __x;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip)
  _LIBCUDACXX_REQUIRES(bidirectional_iterator<_Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Ip
  operator()(_Ip __x, iter_difference_t<_Ip> __n, _Ip __bound_iter) const
  {
    _CUDA_VRANGES::advance(__x, -__n, __bound_iter);
    return __x;
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto prev = __prev::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER > 2014 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___ITERATOR_PREV_H
