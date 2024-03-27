//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_ITERATOR_CONCEPT_H
#define _LIBCUDACXX___ALGORITHM_RANGES_ITERATOR_CONCEPT_H

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

#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/remove_cvref.h"

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _IterMaybeQualified>
_LIBCUDACXX_INLINE_VISIBILITY constexpr auto __get_iterator_concept() {
  using _Iter = __remove_cvref_t<_IterMaybeQualified>;

  if constexpr (contiguous_iterator<_Iter>)
    return contiguous_iterator_tag();
  else if constexpr (random_access_iterator<_Iter>)
    return random_access_iterator_tag();
  else if constexpr (bidirectional_iterator<_Iter>)
    return bidirectional_iterator_tag();
  else if constexpr (forward_iterator<_Iter>)
    return forward_iterator_tag();
  else if constexpr (input_iterator<_Iter>)
    return input_iterator_tag();
}

template <class _Iter>
using __iterator_concept = decltype(__get_iterator_concept<_Iter>());

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___ALGORITHM_RANGES_ITERATOR_CONCEPT_H
