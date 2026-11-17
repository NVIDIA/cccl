//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// SIMD host/device functions are not supported in tile mode.

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <int N>
struct constant_range
{
  int values[N]{};

  constant_range() = delete;
  TEST_FUNC constexpr explicit constant_range(int first)
  {
    for (int i = 0; i < N; ++i)
    {
      values[i] = first + i;
    }
  }
  TEST_FUNC constexpr int* begin()
  {
    return values;
  }
  TEST_FUNC constexpr const int* begin() const
  {
    return values;
  }
  TEST_FUNC constexpr int* end()
  {
    return values + N;
  }
  TEST_FUNC constexpr const int* end() const
  {
    return values + N;
  }
  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return N;
  }
};

struct misleading_tuple : constant_range<3>
{};

struct static_member_range : constant_range<4>
{
  TEST_FUNC static constexpr cuda::std::size_t size()
  {
    return 4;
  }
};

struct vector_wrapper
{
  simd::vec<int, 4> value;

  TEST_FUNC constexpr explicit operator simd::vec<int, 4>() const
  {
    return value;
  }
};

TEST_FUNC constexpr void test_participation()
{
  using Vec = simd::vec<int, 4>;

  // A non-static vector member must not be mistaken for a constexpr wrapper's static value.
  static_assert(cuda::std::is_constructible_v<Vec, vector_wrapper>);
  static_assert(!cuda::std::is_convertible_v<vector_wrapper, Vec>);

  static_assert(cuda::std::is_constructible_v<Vec, int (&)[4]>);
  static_assert(!cuda::std::is_constructible_v<Vec, int (&)[3]>); // different size

  static_assert(cuda::std::is_constructible_v<Vec, constant_range<4>&>);
  static_assert(!cuda::std::is_constructible_v<Vec, constant_range<3>&>); // different size

  static_assert(cuda::std::is_constructible_v<Vec, static_member_range&>);
  static_assert(!cuda::std::is_constructible_v<Vec, cuda::std::span<int>>);
  static_assert(!cuda::std::is_constructible_v<Vec, int*>);
}

template <int N>
struct last_lane
{
  template <class Index>
  TEST_FUNC constexpr bool operator()(Index index) const
  {
    return index == N - 1;
  }
};

template <int N>
TEST_FUNC constexpr void test_values()
{
  using Vec = simd::vec<int, N>;
  int values[N]{};
  for (int i = 0; i < N; ++i)
  {
    values[i] = i + 1;
  }
  const auto& input = values;
  const Vec from_array(input);

  constant_range<N> range(1);
  const Vec from_range(range);

  auto loaded = simd::unchecked_load<Vec>(range);
  for (int i = 0; i < N; ++i)
  {
    assert(from_array[i] == values[i]);
    assert(from_range[i] == values[i]);
    assert(loaded[i] == values[i]);
  }

  constant_range<N> output(0);
  simd::unchecked_store(from_array, output);
  for (int i = 0; i < N; ++i)
  {
    assert(output.values[i] == values[i]);
  }
}

TEST_FUNC constexpr bool test()
{
  test_participation();
  test_values<1>();
  test_values<3>();
  return true;
}

int main(int, char**)
{
  assert(test()); // NOLINT(misc-static-assert)
  static_assert(test());
  return 0;
}
