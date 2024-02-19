// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_INDIRECTLY_COMPARABLE_H
#define _LIBCUDACXX___ITERATOR_INDIRECTLY_COMPARABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__functional/identity.h"
#include "../__iterator/concepts.h"
#include "../__iterator/projected.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2017

template <class _Iter1, class _Iter2, class _BinaryPred,
          class _Proj1 = identity, class _Proj2 = identity>
concept indirectly_comparable =
    indirect_binary_predicate<_BinaryPred, projected<_Iter1, _Proj1>,
                              projected<_Iter2, _Proj2>>;

#elif _CCCL_STD_VER > 2014

// clang-format off

template <class _Iter1, class _Iter2, class _BinaryPred, class _Proj1, class _Proj2>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __indirectly_comparable_,
  requires()(
    requires(indirect_binary_predicate<_BinaryPred, projected<_Iter1, _Proj1>, projected<_Iter2, _Proj2>>)
  ));

template <class _Iter1, class _Iter2, class _BinaryPred, class _Proj1 = identity, class _Proj2 = identity>
_LIBCUDACXX_CONCEPT indirectly_comparable =
  _LIBCUDACXX_FRAGMENT(__indirectly_comparable_, _Iter1, _Iter2, _BinaryPred, _Proj1, _Proj2);

// clang-format on

#endif // _CCCL_STD_VER > 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_INDIRECTLY_COMPARABLE_H
