//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class Src, class Dst>
//   constexpr void copy(const Src& src, const Dst& dst);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct assign_from_int
{
  int value = 0;

  TEST_FUNC constexpr assign_from_int& operator=(int other)
  {
    value = other;
    return *this;
  }
};

struct not_assignable
{
  TEST_FUNC constexpr not_assignable& operator=(const not_assignable&) = delete;
};

template <class Src, class Dst, class = void>
inline constexpr bool can_copy = false;

template <class Src, class Dst>
inline constexpr bool can_copy<
  Src,
  Dst,
  cuda::std::void_t<decltype(cuda::std::copy(cuda::std::declval<const Src&>(), cuda::std::declval<const Dst&>()))>> =
  true;

using extents_1d_2  = cuda::std::extents<int, 2>;
using extents_1d_3  = cuda::std::extents<int, 3>;
using extents_2d    = cuda::std::extents<int, 2, 3>;
using dextents_2d   = cuda::std::extents<int, cuda::std::dynamic_extent, 3>;
using static_mdspan = cuda::std::mdspan<int, extents_2d>;

static_assert(can_copy<static_mdspan, static_mdspan>);
static_assert(can_copy<cuda::std::mdspan<int, dextents_2d>, static_mdspan>);
static_assert(!can_copy<int, static_mdspan>);
static_assert(!can_copy<static_mdspan, int>);
static_assert(!can_copy<cuda::std::mdspan<int, extents_1d_2>, cuda::std::mdspan<int, extents_1d_3>>);
static_assert(!can_copy<cuda::std::mdspan<int, extents_2d>, cuda::std::mdspan<const int, extents_2d>>);
static_assert(
  !can_copy<cuda::std::mdspan<not_assignable, extents_1d_2>, cuda::std::mdspan<not_assignable, extents_1d_2>>);

TEST_FUNC constexpr bool test_static_copy()
{
  cuda::std::array<int, 6> source{1, 2, 3, 4, 5, 6};
  cuda::std::array<int, 6> destination{};

  cuda::std::mdspan<int, extents_2d> source_md(source.data());
  cuda::std::mdspan<int, extents_2d> destination_md(destination.data());

  cuda::std::copy(source_md, destination_md);

  for (int i = 0; i < 6; ++i)
  {
    assert(destination[i] == source[i]);
  }

  return true;
}

TEST_FUNC constexpr bool test_dynamic_source_extents()
{
  cuda::std::array<int, 6> source{6, 5, 4, 3, 2, 1};
  cuda::std::array<int, 6> destination{};

  cuda::std::mdspan<int, dextents_2d> source_md(source.data(), dextents_2d{2});
  cuda::std::mdspan<int, extents_2d> destination_md(destination.data());

  cuda::std::copy(source_md, destination_md);

  for (int i = 0; i < 6; ++i)
  {
    assert(destination[i] == source[i]);
  }

  return true;
}

TEST_FUNC constexpr bool test_mixed_layout_copy()
{
  cuda::std::array<int, 6> source{0, 1, 2, 3, 4, 5};
  cuda::std::array<int, 6> destination{};
  cuda::std::array<int, 2> strides{1, 2};

  cuda::std::layout_stride::mapping<extents_2d> destination_mapping(extents_2d{}, strides);
  cuda::std::mdspan<int, extents_2d> source_md(source.data());
  cuda::std::mdspan<int, extents_2d, cuda::std::layout_stride> destination_md(destination.data(), destination_mapping);

  cuda::std::copy(source_md, destination_md);

  for (int row = 0; row < 2; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      assert(destination_md(row, col) == source_md(row, col));
    }
  }

  return true;
}

TEST_FUNC constexpr bool test_rank_zero_copy()
{
  int source      = 42;
  int destination = 0;

  cuda::std::mdspan<int, cuda::std::extents<int>> source_md(&source);
  cuda::std::mdspan<int, cuda::std::extents<int>> destination_md(&destination);

  cuda::std::copy(source_md, destination_md);

  assert(destination == 42);

  return true;
}

TEST_FUNC constexpr bool test_heterogeneous_copy()
{
  cuda::std::array<int, 3> source{11, 22, 33};
  cuda::std::array<assign_from_int, 3> destination{};

  cuda::std::mdspan<int, extents_1d_3> source_md(source.data());
  cuda::std::mdspan<assign_from_int, extents_1d_3> destination_md(destination.data());

  cuda::std::copy(source_md, destination_md);

  for (int i = 0; i < 3; ++i)
  {
    assert(destination[i].value == source[i]);
  }

  return true;
}

TEST_FUNC constexpr bool test()
{
  test_static_copy();
  test_dynamic_source_extents();
  test_mixed_layout_copy();
  test_rank_zero_copy();
  test_heterogeneous_copy();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
