//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MISMATCH_H
#define _LIBCUDACXX___ALGORITHM_MISMATCH_H

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

#include "../__algorithm/comp.h"
#include "../__iterator/iterator_traits.h"
#include "../__utility/pair.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_InputIterator1, _InputIterator2>
  mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred)
{
  for (; __first1 != __last1; ++__first1, (void) ++__first2)
  {
    if (!__pred(*__first1, *__first2))
    {
      break;
    }
  }
  return pair<_InputIterator1, _InputIterator2>{__first1, __first2};
}

template <class _InputIterator1, class _InputIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_InputIterator1, _InputIterator2>
  mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2)
{
  return _CUDA_VSTD::mismatch(__first1, __last1, __first2, __equal_to{});
}

#if _CCCL_STD_VER > 2011
template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_InputIterator1, _InputIterator2>
  mismatch(_InputIterator1 __first1,
           _InputIterator1 __last1,
           _InputIterator2 __first2,
           _InputIterator2 __last2,
           _BinaryPredicate __pred)
{
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void) ++__first2)
  {
    if (!__pred(*__first1, *__first2))
    {
      break;
    }
  }
  return pair<_InputIterator1, _InputIterator2>{__first1, __first2};
}

template <class _InputIterator1, class _InputIterator2>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<_InputIterator1, _InputIterator2>
  mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2)
{
  return _CUDA_VSTD::mismatch(__first1, __last1, __first2, __last2, __equal_to{});
}
#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_MISMATCH_H
