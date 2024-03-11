//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PARTITION_POINT_H
#define _LIBCUDACXX___ALGORITHM_PARTITION_POINT_H

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

#include "../__algorithm/half_positive.h"
#include "../__iterator/advance.h"
#include "../__iterator/distance.h"
#include "../__iterator/iterator_traits.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Predicate>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
partition_point(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
  typedef typename iterator_traits<_ForwardIterator>::difference_type difference_type;
  difference_type __len = _CUDA_VSTD::distance(__first, __last);
  while (__len != 0)
  {
    difference_type __l2 = _CUDA_VSTD::__half_positive(__len);
    _ForwardIterator __m = __first;
    _CUDA_VSTD::advance(__m, __l2);
    if (__pred(*__m))
    {
      __first = ++__m;
      __len -= __l2 + 1;
    }
    else
    {
      __len = __l2;
    }
  }
  return __first;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_PARTITION_POINT_H
