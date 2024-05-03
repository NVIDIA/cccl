//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FIND_IF_NOT_H
#define _LIBCUDACXX___ALGORITHM_FIND_IF_NOT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Predicate>
_CCCL_NODISCARD inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _InputIterator
find_if_not(_InputIterator __first, _InputIterator __last, _Predicate __pred)
{
  for (; __first != __last; ++__first)
  {
    if (!__pred(*__first))
    {
      break;
    }
  }
  return __first;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_FIND_IF_NOT_H
