//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implement a mechanism to compute the inner part of a shape
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/slice.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

namespace cuda::experimental::stf
{
#ifndef _CCCL_DOXYGEN_INVOKED // doxygen fails to parse this

/**
 * @brief Applying "inner" on a mdspan shape returns an explicit shape which extents
 * have been diminished by a "thickness" constant.
 *
 * For example, a applying inner<2> on mdspan of dimension {M, N} will produce
 * an explicit shape ({2, M-2}, {2, N-2})
 *
 * @tparam thickness number of elements to remove in every direction
 * @tparam rank dimension of the box
 * @param s input box shape
 *
 */
template <size_t thickness, typename T, typename... P>
_CCCL_HOST_DEVICE box<mdspan<T, P...>::rank()> inner(const shape_of<mdspan<T, P...>>& s)
{
  using m               = mdspan<T, P...>;
  constexpr size_t rank = m::rank();

  const ::std::array<size_t, rank> sizes = s.get_sizes();

  ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, rank> inner_extents;
  for (size_t i = 0; i < rank; i++)
  {
    inner_extents[i].first  = thickness;
    inner_extents[i].second = sizes[i] - thickness;
  }

  return box(inner_extents);
}

/**
 * @brief Applying "inner" on an explicit shape returns another explicit shape which
 * extents have been diminished by a "thickness" constant.
 *
 * @overload
 *
 * For example, a applying inner<2> on an explicit shape {{10, 100}, {-10, 10}}
 * will produce the explicit shape ({12, 98}, {-8, 8})
 *
 * @tparam thickness number of elements to remove in every direction
 * @tparam rank dimension of the box
 * @param s input box shape
 */
template <size_t thickness, size_t rank>
_CCCL_HOST_DEVICE box<rank> inner(const box<rank>& s)
{
  ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, rank> inner_extents;
  for (size_t i = 0; i < rank; i++)
  {
    inner_extents[i].first  = s.get_begin(i) + thickness;
    inner_extents[i].second = s.get_end(i) - thickness;
  }

  return box(inner_extents);
}

#endif // _CCCL_DOXYGEN_INVOKED

#ifdef UNITTESTED_FILE
UNITTEST("inner explicit shape (explicit bounds)")
{
  box s({10, 100}, {-10, 10});
  static_assert(::std::is_same_v<decltype(s), box<2>>);

  auto i = inner<2>(s);
  EXPECT(i.get_begin(0) == 12);
  EXPECT(i.get_end(0) == 98);
  EXPECT(i.get_begin(1) == -8);
  EXPECT(i.get_end(1) == 8);
};

UNITTEST("inner explicit shape (sizes)")
{
  box s(10, 100, 12);
  static_assert(::std::is_same_v<decltype(s), box<3>>);

  auto i = inner<2>(s);
  EXPECT(i.get_begin(0) == 2);
  EXPECT(i.get_end(0) == 8);
  EXPECT(i.get_begin(1) == 2);
  EXPECT(i.get_end(1) == 98);
  EXPECT(i.get_begin(2) == 2);
  EXPECT(i.get_end(2) == 10);
};

UNITTEST("inner mdspan shape")
{
  auto s = shape_of<slice<double, 3>>(10, 100, 12);

  auto i = inner<2>(s);
  EXPECT(i.get_begin(0) == 2);
  EXPECT(i.get_end(0) == 8);
  EXPECT(i.get_begin(1) == 2);
  EXPECT(i.get_end(1) == 98);
  EXPECT(i.get_begin(2) == 2);
  EXPECT(i.get_end(2) == 10);
};

#endif // UNITTESTED_FILE
} // namespace cuda::experimental::stf
