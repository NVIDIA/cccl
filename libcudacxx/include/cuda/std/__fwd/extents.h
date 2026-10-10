//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_EXTENTS_H
#define _CUDA_STD___FWD_EXTENTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _IndexType, size_t... _Extents>
class extents;

template <class _Tp>
inline constexpr bool __is_cuda_std_extents_v = false;
template <class _IndexType, size_t... _Extents>
inline constexpr bool __is_cuda_std_extents_v<extents<_IndexType, _Extents...>> = true;

// Recursive helper classes to implement dextents alias for extents
template <class _IndexType, size_t _Rank, class _Extents = extents<_IndexType>>
struct __make_dextents;

template <class _IndexType, size_t _Rank, class _Extents = extents<_IndexType>>
using __make_dextents_t = typename __make_dextents<_IndexType, _Rank, _Extents>::type;

template <class _IndexType, size_t _Rank, size_t... _ExtentsPack>
struct __make_dextents<_IndexType, _Rank, extents<_IndexType, _ExtentsPack...>>
{
  using type = __make_dextents_t<_IndexType, _Rank - 1, extents<_IndexType, dynamic_extent, _ExtentsPack...>>;
};

template <class _IndexType, size_t... _ExtentsPack>
struct __make_dextents<_IndexType, 0, extents<_IndexType, _ExtentsPack...>>
{
  using type = extents<_IndexType, _ExtentsPack...>;
};

// [mdspan.extents.dextents], alias template
template <class _IndexType, size_t _Rank>
using dextents = __make_dextents_t<_IndexType, _Rank>;

template <size_t _Rank, class _IndexType = size_t>
using dims = dextents<_IndexType, _Rank>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_MDSPAN_H
