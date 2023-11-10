// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_NEXT_H
#define _LIBCUDACXX___ITERATOR_NEXT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__iterator/advance.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/enable_if.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    __enable_if_t<__is_cpp17_input_iterator<_InputIter>::value, _InputIter>
    next(_InputIter __x, typename iterator_traits<_InputIter>::difference_type __n = 1) {
  _LIBCUDACXX_ASSERT(__n >= 0 || __is_cpp17_bidirectional_iterator<_InputIter>::value,
                 "Attempt to next(it, n) with negative n on a non-bidirectional iterator");

  _CUDA_VSTD::advance(__x, __n);
  return __x;
}

_LIBCUDACXX_END_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14 && !defined(_LIBCUDACXX_COMPILER_MSVC_2017)

// [range.iter.op.next]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__next)
struct __fn {
  _LIBCUDACXX_TEMPLATE(class _Ip)
    _LIBCUDACXX_REQUIRES(input_or_output_iterator<_Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Ip operator()(_Ip __x) const {
    ++__x;
    return __x;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip)
    _LIBCUDACXX_REQUIRES(input_or_output_iterator<_Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n) const {
    _CUDA_VRANGES::advance(__x, __n);
    return __x;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp)
    _LIBCUDACXX_REQUIRES(input_or_output_iterator<_Ip> && sentinel_for<_Sp, _Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Ip operator()(_Ip __x, _Sp __bound_sentinel) const {
    _CUDA_VRANGES::advance(__x, __bound_sentinel);
    return __x;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp)
    _LIBCUDACXX_REQUIRES(input_or_output_iterator<_Ip> && sentinel_for<_Sp, _Ip>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n, _Sp __bound_sentinel) const {
    _CUDA_VRANGES::advance(__x, __n, __bound_sentinel);
    return __x;
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto next = __next::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX_STD_VER > 14 && !defined(_LIBCUDACXX_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___ITERATOR_NEXT_H
