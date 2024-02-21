//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_PUSH_HEAP_H
#define _LIBCUDACXX___ALGORITHM_PUSH_HEAP_H

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
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__utility/move.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
__sift_up(_RandomAccessIterator __first,
          _RandomAccessIterator __last,
          _Compare&& __comp,
          typename iterator_traits<_RandomAccessIterator>::difference_type __len) {
  using value_type = typename iterator_traits<_RandomAccessIterator>::value_type;

  if (__len > 1) {
    __len                       = (__len - 2) / 2;
    _RandomAccessIterator __ptr = __first + __len;

    if (__comp(*__ptr, *--__last)) {
      value_type __t(_IterOps<_AlgPolicy>::__iter_move(__last));
      do {
        *__last = _IterOps<_AlgPolicy>::__iter_move(__ptr);
        __last  = __ptr;
        if (__len == 0)
          break;
        __len = (__len - 1) / 2;
        __ptr = __first + __len;
      } while (__comp(*__ptr, __t));

      *__last = _CUDA_VSTD::move(__t);
    }
  }
}

template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
__push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare& __comp) {
  typename iterator_traits<_RandomAccessIterator>::difference_type __len = __last - __first;
  _CUDA_VSTD::__sift_up<_AlgPolicy, __comp_ref_type<_Compare> >(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp, __len);
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  static_assert(_CUDA_VSTD::is_copy_constructible<_RandomAccessIterator>::value, "Iterators must be copy constructible.");
  static_assert(_CUDA_VSTD::is_copy_assignable<_RandomAccessIterator>::value, "Iterators must be copy assignable.");

  _CUDA_VSTD::__push_heap<_ClassicAlgPolicy>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 void
push_heap(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  _CUDA_VSTD::push_heap(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_PUSH_HEAP_H
