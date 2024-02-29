//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MAX_ELEMENT_H
#define _LIBCUDACXX___ALGORITHM_MAX_ELEMENT_H

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
#include "../__algorithm/comp_ref_type.h"
#include "../__iterator/iterator_traits.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Compare, class _ForwardIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
__max_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  static_assert(__is_cpp17_input_iterator<_ForwardIterator>::value,
                "_CUDA_VSTD::max_element requires a ForwardIterator");
  if (__first != __last)
  {
    _ForwardIterator __i = __first;
    while (++__i != __last)
    {
      if (__comp(*__first, *__i))
      {
        __first = __i;
      }
    }
  }
  return __first;
}

template <class _ForwardIterator, class _Compare>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
max_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  return _CUDA_VSTD::__max_element<__comp_ref_type<_Compare> >(__first, __last, __comp);
}

template <class _ForwardIterator>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
max_element(_ForwardIterator __first, _ForwardIterator __last)
{
  return _CUDA_VSTD::max_element(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_MAX_ELEMENT_H
