//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_REVERSE_COPY_H
#define _LIBCUDACXX___ALGORITHM_REVERSE_COPY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _BidirectionalIterator, class _OutputIterator>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _OutputIterator
reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __result)
{
  for (; __first != __last; ++__result)
  {
    *__result = *--__last;
  }
  return __result;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_REVERSE_COPY_H
