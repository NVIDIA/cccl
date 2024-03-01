//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SORT_HEAP_H
#define _LIBCUDACXX___ALGORITHM_SORT_HEAP_H

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
#include "../__algorithm/iterator_operations.h"
#include "../__algorithm/pop_heap.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__utility/move.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
__sort_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp)
{
  __comp_ref_type<_Compare> __comp_ref = __comp;

  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  for (difference_type __n = __last - __first; __n > 1; --__last, (void) --__n)
  {
    _CUDA_VSTD::__pop_heap<_AlgPolicy>(__first, __last, __comp_ref, __n);
  }
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
sort_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
  static_assert(_LIBCUDACXX_TRAIT(is_copy_constructible, _RandomAccessIterator),
                "Iterators must be copy constructible.");
  static_assert(_LIBCUDACXX_TRAIT(is_copy_assignable, _RandomAccessIterator), "Iterators must be copy assignable.");

  _CUDA_VSTD::__sort_heap<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
sort_heap(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
  _CUDA_VSTD::sort_heap(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_SORT_HEAP_H
