// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_TRANSFORM_INCLUSIVE_SCAN_H
#define _LIBCUDACXX___NUMERIC_TRANSFORM_INCLUSIVE_SCAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp, class _UnaryOp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _OutputIterator transform_inclusive_scan(
  _InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryOp __b, _UnaryOp __u, _Tp __init)
{
  for (; __first != __last; ++__first, (void) ++__result)
  {
    __init    = __b(__init, __u(*__first));
    *__result = __init;
  }

  return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOp, class _UnaryOp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _OutputIterator transform_inclusive_scan(
  _InputIterator __first, _InputIterator __last, _OutputIterator __result, _BinaryOp __b, _UnaryOp __u)
{
  if (__first != __last)
  {
    typename iterator_traits<_InputIterator>::value_type __init = __u(*__first);
    *__result++                                                 = __init;
    if (++__first != __last)
    {
      return _CUDA_VSTD::transform_inclusive_scan(__first, __last, __result, __b, __u, __init);
    }
  }

  return __result;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NUMERIC_TRANSFORM_INCLUSIVE_SCAN_H
