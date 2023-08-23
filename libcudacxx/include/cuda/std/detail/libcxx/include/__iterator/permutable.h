// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_PERMUTABLE_H
#define _LIBCUDACXX___ITERATOR_PERMUTABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"
#include "../__iterator/iter_swap.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

template <class _Iterator>
concept permutable =
    forward_iterator<_Iterator> &&
    indirectly_movable_storable<_Iterator, _Iterator> &&
    indirectly_swappable<_Iterator, _Iterator>;

#elif _LIBCUDACXX_STD_VER > 14

template<class _Iterator>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __permutable_,
  requires()(
    requires(forward_iterator<_Iterator>),
    requires(indirectly_movable_storable<_Iterator, _Iterator>),
    requires(indirectly_swappable<_Iterator, _Iterator>)
  ));

template <class _Iterator>
_LIBCUDACXX_CONCEPT permutable = _LIBCUDACXX_FRAGMENT(__permutable_, _Iterator);

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_PERMUTABLE_H
