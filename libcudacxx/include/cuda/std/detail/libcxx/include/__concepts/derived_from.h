//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_DERIVED_FROM_H
#define _LIBCUDACXX___CONCEPTS_DERIVED_FROM_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/add_pointer.h"
#include "../__type_traits/is_base_of.h"
#include "../__type_traits/is_convertible.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2017

// [concept.derived]

template<class _Dp, class _Bp>
concept derived_from =
  is_base_of_v<_Bp, _Dp> &&
  is_convertible_v<const volatile _Dp*, const volatile _Bp*>;

#elif _CCCL_STD_VER > 2011

template<class _Dp, class _Bp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __derived_from_,
  requires()(
    requires(_LIBCUDACXX_TRAIT(is_base_of, _Bp, _Dp)),
    requires(_LIBCUDACXX_TRAIT(is_convertible, add_pointer_t<const volatile _Dp>, add_pointer_t<const volatile _Bp>))
  ));

template<class _Dp, class _Bp>
_LIBCUDACXX_CONCEPT derived_from = _LIBCUDACXX_FRAGMENT(__derived_from_, _Dp, _Bp);

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_DERIVED_FROM_H
