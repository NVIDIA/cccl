//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_REPLACE_COPY_IF_H
#define _LIBCUDACXX___ALGORITHM_REPLACE_COPY_IF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIterator, class _OutputIterator, class _Predicate, class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _OutputIterator replace_copy_if(
  _InputIterator __first, _InputIterator __last, _OutputIterator __result, _Predicate __pred, const _Tp& __new_value)
{
  for (; __first != __last; ++__first, (void) ++__result)
  {
    if (__pred(*__first))
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

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_REPLACE_COPY_IF_H
