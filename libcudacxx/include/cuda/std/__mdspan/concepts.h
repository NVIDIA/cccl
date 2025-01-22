// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_CONCEPTS_H
#define _LIBCUDACXX___MDSPAN_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __mdspan_detail
{

// [mdspan.layout.stride.expo]/3
template <class>
struct __is_extents : false_type
{};

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_extents_v = __is_extents<_Tp>::value;

// [mdspan.layout.general]/2
template <class _Layout, class _Mapping>
_CCCL_INLINE_VAR constexpr bool __is_mapping_of =
  _CCCL_TRAIT(is_same, typename _Layout::template mapping<typename _Mapping::extents_type>, _Mapping);

// [mdspan.layout.stride.expo]/4
#if _CCCL_STD_VER >= 2020
template <class _Mapping>
concept __layout_mapping_alike = requires {
  requires __is_mapping_of<typename _Mapping::layout_type, _Mapping>;
  requires __is_extents_v<typename _Mapping::extents_type>;
  { _Mapping::is_always_strided() } -> same_as<bool>;
  { _Mapping::is_always_exhaustive() } -> same_as<bool>;
  { _Mapping::is_always_unique() } -> same_as<bool>;
  bool_constant<_Mapping::is_always_strided()>::value;
  bool_constant<_Mapping::is_always_exhaustive()>::value;
  bool_constant<_Mapping::is_always_unique()>::value;
};
#else // ^^^ _CCCL_STD_VER >= 2020 ^^^ / vvv _CCCL_STD_VER <= 2017 vvv
// NOTE: integral_constant<bool, _Mapping::is_always_strided()>::value only checks that this is a constant expression
template <class _Mapping>
_CCCL_CONCEPT_FRAGMENT(
  __layout_mapping_alike_prop_,
  requires()( //
    requires(same_as<bool, decltype(_Mapping::is_always_strided())>),
    requires(same_as<bool, decltype(_Mapping::is_always_exhaustive())>),
    requires(same_as<bool, decltype(_Mapping::is_always_unique())>),
    (integral_constant<bool, _Mapping::is_always_strided()>::value),
    (integral_constant<bool, _Mapping::is_always_exhaustive()>::value),
    (integral_constant<bool, _Mapping::is_always_unique()>::value)));

template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_alike_prop = _CCCL_FRAGMENT(__layout_mapping_alike_prop_, _Mapping);

template <class _Mapping>
_CCCL_CONCEPT_FRAGMENT(
  __layout_mapping_alike_,
  requires()( //
    requires(__is_mapping_of<typename _Mapping::layout_type, _Mapping>),
    requires(_CCCL_TRAIT(__is_extents, typename _Mapping::extents_type)),
    requires(__layout_mapping_alike_prop<_Mapping>)));

template <class _Mapping>
_CCCL_CONCEPT __layout_mapping_alike = _CCCL_FRAGMENT(__layout_mapping_alike_, _Mapping);
#endif // _CCCL_STD_VER <= 2017

} // namespace __mdspan_detail

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MDSPAN_CONCEPTS_H
