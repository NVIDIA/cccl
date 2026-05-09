//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  // Uniform scalar via CTAD
  {
    auto da = cuda::dynamic_argument{5};
    assert(da.arg == 5);
    static_assert(cuda::argument_traits<decltype(da)>::lowest == cuda::std::numeric_limits<int>::lowest());
    static_assert(cuda::argument_traits<decltype(da)>::max == cuda::std::numeric_limits<int>::max());
    assert(cuda::argument_lowest(da) == cuda::std::numeric_limits<int>::lowest());
    assert(cuda::argument_max(da) == cuda::std::numeric_limits<int>::max());
  }

  // Uniform scalar with static bounds
  {
    auto da = cuda::dynamic_argument{5, cuda::argument_bounds<1, 8>()};
    assert(da.arg == 5);
    static_assert(cuda::argument_traits<decltype(da)>::lowest == 1);
    static_assert(cuda::argument_traits<decltype(da)>::max == 8);
  }

  // Per-segment span with runtime bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto da    = cuda::dynamic_argument{cuda::std::span<int>{arr, 4}, cuda::argument_bounds(1, 100)};
    assert(da.arg.size() == 4);
    assert(cuda::argument_lowest(da) == 1);
    assert(cuda::argument_max(da) == 100);
  }

  // Per-segment span with both bounds — argument_max combines both
  {
    int arr[4] = {10, 20, 30, 40};
    auto da    = cuda::dynamic_argument{
      cuda::std::span<int>{arr, 4}, cuda::argument_bounds<1, 256>(), cuda::argument_bounds(10, 200)};
    static_assert(cuda::argument_traits<decltype(da)>::lowest == 1);
    static_assert(cuda::argument_traits<decltype(da)>::max == 256);
    assert(cuda::argument_lowest(da) == 10);
    assert(cuda::argument_max(da) == 200);
  }

  // Per-segment via span
  {
    int arr[4] = {1, 2, 3, 4};
    auto da    = cuda::dynamic_argument{cuda::std::span<int>{arr, 4}};
    assert(da.arg.size() == 4);
    assert(da.arg[0] == 1);
    assert(da.arg[3] == 4);
  }

  // Per-segment with static bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto da    = cuda::dynamic_argument{cuda::std::span<int>{arr, 4}, cuda::argument_bounds<1, 100>()};
    assert(da.arg.size() == 4);
    assert(da.arg[2] == 30);
    static_assert(cuda::argument_traits<decltype(da)>::lowest == 1);
    static_assert(cuda::argument_traits<decltype(da)>::max == 100);
  }

  // Uniform via span<T, 1> (span<T, 1>)
  {
    int val = 42;
    auto da = cuda::dynamic_argument{cuda::std::span<int, 1>{&val, 1}};
    assert(da.arg[0] == 42);
  }

  // Traits
  {
    using traits = cuda::argument_traits<cuda::dynamic_argument<int>>;
    static_assert(!traits::is_deferred);
    static_assert(cuda::std::is_same_v<traits::value_type, int>);
  }

  // __is_single_value_v on unwrapped types
  {
    static_assert(cuda::__is_single_value_v<cuda::argument_traits<cuda::dynamic_argument<int>>::value_type>);
    static_assert(
      !cuda::__is_single_value_v<cuda::argument_traits<cuda::dynamic_argument<cuda::std::span<int>>>::value_type>);
    static_assert(
      cuda::__is_single_value_v<cuda::argument_traits<cuda::dynamic_argument<cuda::std::span<int, 1>>>::value_type>);
  }

  // Unwrap: scalar
  {
    auto da       = cuda::dynamic_argument{7};
    const auto& v = cuda::unwrap_argument(da);
    assert(v == 7);
  }

  // Unwrap: span
  {
    int arr[3]    = {10, 20, 30};
    auto da       = cuda::dynamic_argument{cuda::std::span<int>{arr, 3}};
    const auto& v = cuda::unwrap_argument(da);
    assert(v.size() == 3);
    assert(v[1] == 20);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
