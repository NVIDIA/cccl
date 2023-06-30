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

#include "../__iterator/concepts.h"
#include "../__iterator/projected.h"
#include "../__functional/identity.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

template <class _I1, class _I2, class _Rp, class _P1 = identity, class _P2 = identity>
concept indirectly_comparable =
  indirect_binary_predicate<_Rp, projected<_I1, _P1>, projected<_I2, _P2>>;

#elif _LIBCUDACXX_STD_VER > 14

template <class _I1, class _I2, class _Rp, class _P1, class _P2>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __indirectly_comparable_,
  requires()(
    requires(indirect_binary_predicate<_Rp, projected<_I1, _P1>, projected<_I2, _P2>>)
  ));

template <class _I1, class _I2, class _Rp, class _P1 = identity, class _P2 = identity>
_LIBCUDACXX_CONCEPT indirectly_comparable =
  _LIBCUDACXX_FRAGMENT(__indirectly_comparable_, _I1, _I2, _Rp, _P1, _P2);

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_INDIRECTLY_COMPARABLE_H
