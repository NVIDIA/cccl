// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_SORTABLE_H
#define _LIBCUDACXX___ITERATOR_SORTABLE_H

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
#include "../__functional/ranges_operations.h"
#include "../__iterator/concepts.h"
#include "../__iterator/permutable.h"
#include "../__iterator/projected.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2017

template <class _Iter, class _Comp = _CUDA_VRANGES::less, class _Proj = identity>
concept sortable =
  permutable<_Iter> &&
  indirect_strict_weak_order<_Comp, projected<_Iter, _Proj>>;

#elif _CCCL_STD_VER > 2014

template <class _Iter, class _Comp, class _Proj>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __sortable_,
  requires()( //
    requires(permutable<_Iter>),
    requires(indirect_strict_weak_order<_Comp, projected<_Iter, _Proj>>)
  ));

template <class _Iter, class _Comp = _CUDA_VRANGES::less, class _Proj = identity>
_LIBCUDACXX_CONCEPT sortable = _LIBCUDACXX_FRAGMENT(__sortable_, _Iter, _Comp, _Proj);

#endif // _CCCL_STD_VER > 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_SORTABLE_H
