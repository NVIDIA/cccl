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

#ifndef _LIBCUDACXX___FWD_MDSPAN_H
#define _LIBCUDACXX___FWD_MDSPAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/void_t.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Layout policy with a mapping which corresponds to FORTRAN-style array layouts
struct layout_left
{
  template <class _Extents>
  class mapping;
};

// Layout policy with a mapping which corresponds to C-style array layouts
struct layout_right
{
  template <class _Extents>
  class mapping;
};

// Layout policy with a unique mapping where strides are arbitrary
struct layout_stride
{
  template <class Extents>
  class mapping;
};

// [mdspan.layout.policy.reqmts]
namespace __mdspan_detail
{
template <class _Layout, class _Extents, class = void>
_CCCL_INLINE_VAR constexpr bool __is_valid_layout_mapping = false;

template <class _Layout, class _Extents>
_CCCL_INLINE_VAR constexpr bool
  __is_valid_layout_mapping<_Layout, _Extents, void_t<typename _Layout::template mapping<_Extents>>> = true;
} // namespace __mdspan_detail

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_MDSPAN_H
