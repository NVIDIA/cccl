//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_REPLACE_COPY_H
#define _LIBCUDACXX___ALGORITHM_REPLACE_COPY_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _OutputIterator, class _Tp>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _OutputIterator replace_copy(
  _InputIterator __first,
  _InputIterator __last,
  _OutputIterator __result,
  const _Tp& __old_value,
  const _Tp& __new_value)
{
  for (; __first != __last; ++__first, (void) ++__result)
  {
    if (*__first == __old_value)
    {
      *__result = __new_value;
    }
    else
    {
      *__result = *__first;
    }
  }
  return __result;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_REPLACE_COPY_H
