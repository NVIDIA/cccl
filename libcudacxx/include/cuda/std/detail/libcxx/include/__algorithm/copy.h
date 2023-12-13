//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COPY_H
#define _LIBCUDACXX___ALGORITHM_COPY_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#include "../__algorithm/unwrap_iter.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constant_evaluated.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_trivially_copy_assignable.h"
#include "../__type_traits/remove_const.h"
#include "../cstdlib"
#include "../cstring"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _OutputIterator>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _OutputIterator
__copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  for (; __first != __last; ++__first, (void) ++__result)
  {
    *__result = *__first;
  }
  return __result;
}

template <class _Tp, class _Up>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 bool
__dispatch_memmove(_Up* __result, _Tp* __first, const size_t __n)
{
#if _CCCL_STD_VER >= 2020
  if (_CUDA_VSTD::is_constant_evaluated())
  {
    return false;
  }
  else
#endif // _CCCL_STD_VER >= 2020
  {
    // For now, we only ever use memmove on host
    // clang-format off
    NV_IF_ELSE_TARGET(NV_IS_HOST, (
      _CUDA_VSTD::memmove(__result, __first, __n * sizeof(_Up));
      return true;
    ),(
      return false;
    ))
    // clang-format on
  }
}

template <class _Tp,
          class _Up,
          __enable_if_t<_LIBCUDACXX_TRAIT(is_same, __remove_const_t<_Tp>, _Up), int> = 0,
          __enable_if_t<_LIBCUDACXX_TRAIT(is_trivially_copy_assignable, _Up), int>   = 0>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _Up*
__copy(_Tp* __first, _Tp* __last, _Up* __result)
{
  const ptrdiff_t __n = __last - __first;
  if (__n > 0)
  {
    if (__dispatch_memmove(__result, __first, __n))
    {
      return __result + __n;
    }
    for (ptrdiff_t __i = 0; __i < __n; ++__i)
    {
      *(__result + __i) = *(__first + __i);
    }
  }
  return __result + __n;
}

template <class _InputIterator, class _OutputIterator>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _OutputIterator
copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  return _CUDA_VSTD::__copy(__unwrap_iter(__first), __unwrap_iter(__last), __unwrap_iter(__result));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_COPY_H
