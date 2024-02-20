//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COPY_N_H
#define _LIBCUDACXX___ALGORITHM_COPY_N_H

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

#include "../__algorithm/copy.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/enable_if.h"
#include "../__utility/convert_to_integral.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator,
          class _Size,
          class _OutputIterator,
          __enable_if_t<__is_cpp17_input_iterator<_InputIterator>::value, int>          = 0,
          __enable_if_t<!__is_cpp17_random_access_iterator<_InputIterator>::value, int> = 0>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _OutputIterator
copy_n(_InputIterator __first, _Size __orig_n, _OutputIterator __result)
{
  using _IntegralSize = decltype(__convert_to_integral(__orig_n));
  _IntegralSize __n   = static_cast<_IntegralSize>(__orig_n);
  if (__n > 0)
  {
    *__result = *__first;
    ++__result;
    for (--__n; __n > 0; --__n)
    {
      ++__first;
      *__result = *__first;
      ++__result;
    }
  }
  return __result;
}

template <class _InputIterator,
          class _Size,
          class _OutputIterator,
          __enable_if_t<__is_cpp17_random_access_iterator<_InputIterator>::value, int> = 0>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _OutputIterator
copy_n(_InputIterator __first, _Size __orig_n, _OutputIterator __result)
{
  using _IntegralSize = decltype(__convert_to_integral(__orig_n));
  _IntegralSize __n   = static_cast<_IntegralSize>(__orig_n);
  return _CUDA_VSTD::copy(__first, __first + __n, __result);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_COPY_N_H
