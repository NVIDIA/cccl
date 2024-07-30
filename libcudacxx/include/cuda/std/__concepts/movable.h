//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_MOVABLE_H
#define _LIBCUDACXX___CONCEPTS_MOVABLE_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/swappable.h>
#include <cuda/std/__type_traits/is_object.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2017

template <class _Tp>
concept movable = is_object_v<_Tp> && move_constructible<_Tp> && assignable_from<_Tp&, _Tp> && swappable<_Tp>;

#elif _CCCL_STD_VER > 2011

// [concepts.object]
template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  _Movable_,
  requires()(requires(is_object_v<_Tp>),
             requires(move_constructible<_Tp>),
             requires(assignable_from<_Tp&, _Tp>),
             requires(swappable<_Tp>)));

template <class _Tp>
_LIBCUDACXX_CONCEPT movable = _LIBCUDACXX_FRAGMENT(_Movable_, _Tp);

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_MOVABLE_H
