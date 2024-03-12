//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_EQUAL_H
#define _LIBCUDACXX___ALGORITHM_EQUAL_H

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
#include "../__iterator/distance.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/add_lvalue_reference.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred)
{
  for (; __first1 != __last1; ++__first1, (void) ++__first2)
  {
    if (!__pred(*__first1, *__first2))
    {
      return false;
    }
  }
  return true;
}

template <class _InputIterator1, class _InputIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2)
{
  return _CUDA_VSTD::equal(__first1, __last1, __first2, __equal_to{});
}

template <class _BinaryPredicate, class _InputIterator1, class _InputIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
__equal(_InputIterator1 __first1,
        _InputIterator1 __last1,
        _InputIterator2 __first2,
        _InputIterator2 __last2,
        _BinaryPredicate __pred,
        input_iterator_tag,
        input_iterator_tag)
{
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void) ++__first2)
  {
    if (!__pred(*__first1, *__first2))
    {
      return false;
    }
  }
  return __first1 == __last1 && __first2 == __last2;
}

template <class _BinaryPredicate, class _RandomAccessIterator1, class _RandomAccessIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
__equal(_RandomAccessIterator1 __first1,
        _RandomAccessIterator1 __last1,
        _RandomAccessIterator2 __first2,
        _RandomAccessIterator2 __last2,
        _BinaryPredicate __pred,
        random_access_iterator_tag,
        random_access_iterator_tag)
{
  if (__last1 - __first1 != __last2 - __first2)
  {
    return false;
  }
  return _CUDA_VSTD::equal<_RandomAccessIterator1, _RandomAccessIterator2, __add_lvalue_reference_t<_BinaryPredicate>>(
    __first1, __last1, __first2, __pred);
}

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
equal(_InputIterator1 __first1,
      _InputIterator1 __last1,
      _InputIterator2 __first2,
      _InputIterator2 __last2,
      _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__equal<__add_lvalue_reference_t<_BinaryPredicate>>(
    __first1,
    __last1,
    __first2,
    __last2,
    __pred,
    typename iterator_traits<_InputIterator1>::iterator_category(),
    typename iterator_traits<_InputIterator2>::iterator_category());
}

template <class _InputIterator1, class _InputIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
equal(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2)
{
  return _CUDA_VSTD::__equal(
    __first1,
    __last1,
    __first2,
    __last2,
    __equal_to{},
    typename iterator_traits<_InputIterator1>::iterator_category(),
    typename iterator_traits<_InputIterator2>::iterator_category());
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_EQUAL_H
