//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_BINARY_SEARCH_H
#define _LIBCUDACXX___ALGORITHM_BINARY_SEARCH_H

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
#include "../__algorithm/lower_bound.h"
#include "../__iterator/iterator_traits.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Tp, class _Compare>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
binary_search(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp) {
  __first = _CUDA_VSTD::lower_bound<_ForwardIterator, _Tp, __comp_ref_type<_Compare> >(__first, __last, __value, __comp);
  return __first != __last && !__comp(__value, *__first);
}

template <class _ForwardIterator, class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
binary_search(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  return _CUDA_VSTD::binary_search(__first, __last, __value, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_BINARY_SEARCH_H
