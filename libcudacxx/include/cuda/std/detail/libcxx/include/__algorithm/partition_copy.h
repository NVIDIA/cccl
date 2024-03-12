//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PARTITION_COPY_H
#define _LIBCUDACXX___ALGORITHM_PARTITION_COPY_H

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

#include "../__iterator/iterator_traits.h"
#include "../__utility/pair.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _OutputIterator1, class _OutputIterator2, class _Predicate>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair<_OutputIterator1, _OutputIterator2> partition_copy(
  _InputIterator __first,
  _InputIterator __last,
  _OutputIterator1 __out_true,
  _OutputIterator2 __out_false,
  _Predicate __pred)
{
  for (; __first != __last; ++__first)
  {
    if (__pred(*__first))
    {
      *__out_true = *__first;
      ++__out_true;
    }
    else
    {
      *__out_false = *__first;
      ++__out_false;
    }
  }
  return pair<_OutputIterator1, _OutputIterator2>(__out_true, __out_false);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_PARTITION_COPY_H
