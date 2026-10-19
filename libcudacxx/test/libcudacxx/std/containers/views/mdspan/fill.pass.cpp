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

// template<class Dst, class T = typename Dst::value_type>
//   constexpr void fill(const Dst& dst, const T& value);

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

template <class Dst, class T, class = void>
inline constexpr bool can_fill = false;

template <class Dst, class T>
inline constexpr bool
  can_fill<Dst, T, cuda::std::void_t<decltype(cuda::std::fill(cuda::std::declval<Dst>(), cuda::std::declval<T>()))>> =
    true;

using extents_1d_2  = cuda::std::extents<int, 2>;
using extents_2d    = cuda::std::extents<int, 2, 3>;
using dextents_2d   = cuda::std::extents<int, cuda::std::dynamic_extent, 3>;
using static_mdspan = cuda::std::mdspan<int, extents_2d>;

static_assert(can_fill<static_mdspan, int>);
static_assert(can_fill<static_mdspan, short>);
static_assert(!can_fill<int, int>);
static_assert(!can_fill<cuda::std::mdspan<const int, extents_2d>, int>);
static_assert(!can_fill<cuda::std::mdspan<not_assignable, extents_1d_2>, not_assignable>);

TEST_FUNC constexpr bool test_static_fill()
{
  cuda::std::array<int, 6> values{};
  cuda::std::mdspan<int, extents_2d> md(values.data());

  cuda::std::fill(md, 7);

  for (int i = 0; i < 6; ++i)
  {
    assert(values[i] == 7);
  }

  return true;
}

TEST_FUNC constexpr bool test_dynamic_fill()
{
  cuda::std::array<int, 6> values{};
  cuda::std::mdspan<int, dextents_2d> md(values.data(), dextents_2d{2});

  cuda::std::fill(md, 9);

  for (int i = 0; i < 6; ++i)
  {
    assert(values[i] == 9);
  }

  return true;
}

TEST_FUNC constexpr bool test_layout_stride_fill()
{
  cuda::std::array<int, 6> values{};
  cuda::std::array<int, 2> strides{1, 2};

  cuda::std::layout_stride::mapping<extents_2d> mapping(extents_2d{}, strides);
  cuda::std::mdspan<int, extents_2d, cuda::std::layout_stride> md(values.data(), mapping);

  cuda::std::fill(md, 5);

  for (int row = 0; row < 2; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      assert(md(row, col) == 5);
    }
  }

  return true;
}

TEST_FUNC constexpr bool test_rank_zero_fill()
{
  int value = 0;
  cuda::std::mdspan<int, cuda::std::extents<int>> md(&value);

  cuda::std::fill(md, 42);

  assert(value == 42);

  return true;
}

TEST_FUNC constexpr bool test_heterogeneous_fill()
{
  cuda::std::array<assign_from_int, 3> values{};
  cuda::std::mdspan<assign_from_int, cuda::std::extents<int, 3>> md(values.data());

  cuda::std::fill(md, 17);

  for (int i = 0; i < 3; ++i)
  {
    assert(values[i].value == 17);
  }

  return true;
}

TEST_FUNC constexpr bool test()
{
  test_static_fill();
  test_dynamic_fill();
  test_layout_stride_fill();
  test_rank_zero_fill();
  test_heterogeneous_fill();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
