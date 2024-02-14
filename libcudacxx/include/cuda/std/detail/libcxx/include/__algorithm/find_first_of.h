//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FIND_FIRST_OF_H
#define _LIBCUDACXX___ALGORITHM_FIND_FIRST_OF_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator1
__find_first_of_ce(_ForwardIterator1 __first1,
                   _ForwardIterator1 __last1,
                   _ForwardIterator2 __first2,
                   _ForwardIterator2 __last2,
                   _BinaryPredicate __pred)
{
  for (; __first1 != __last1; ++__first1)
  {
    for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
    {
      if (__pred(*__first1, *__j))
      {
        return __first1;
      }
    }
  }
  return __last1;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator1
find_first_of(_ForwardIterator1 __first1,
              _ForwardIterator1 __last1,
              _ForwardIterator2 __first2,
              _ForwardIterator2 __last2,
              _BinaryPredicate __pred)
{
  return _CUDA_VSTD::__find_first_of_ce(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator1
find_first_of(
  _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2)
{
  return _CUDA_VSTD::__find_first_of_ce(__first1, __last1, __first2, __last2, __equal_to{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_FIND_FIRST_OF_H
