//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_IS_SORTED_UNTIL_H
#define _LIBCUDACXX___ALGORITHM_IS_SORTED_UNTIL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/comp_ref_type.h>
#include <cuda/std/__iterator/iterator_traits.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Compare, class _ForwardIterator>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _ForwardIterator
__is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    while (++__i != __last)
    {
      if (__comp(*__i, *__first))
      {
        return __i;
      }
      __first = __i;
    }
  }
  return __last;
}

template <class _ForwardIterator, class _Compare>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  return _CUDA_VSTD::__is_sorted_until<__comp_ref_type<_Compare>>(__first, __last, __comp);
}

template <class _ForwardIterator>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last)
{
  return _CUDA_VSTD::is_sorted_until(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_IS_SORTED_UNTIL_H
