//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__argument_>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct non_sequence_value
{
  int payload;
};

TEST_FUNC constexpr bool test()
{
  // Uniform scalar via CTAD
  {
    auto da = cuda::__argument::__immediate{5};
    assert(cuda::__argument::__unwrap(da) == 5);
    static_assert(cuda::__argument::__traits<decltype(da)>::lowest == cuda::std::numeric_limits<int>::lowest());
    static_assert(cuda::__argument::__traits<decltype(da)>::highest == (cuda::std::numeric_limits<int>::max)());
    assert(cuda::__argument::__lowest_(da) == 5);
    assert(cuda::__argument::__highest_(da) == 5);
  }

  // Uniform scalar with static bounds
  {
    auto da = cuda::__argument::__immediate{5, cuda::__argument::__bounds<1, 8>()};
    assert(cuda::__argument::__unwrap(da) == 5);
    static_assert(cuda::__argument::__traits<decltype(da)>::lowest == 1);
    static_assert(cuda::__argument::__traits<decltype(da)>::highest == 8);
    assert(cuda::__argument::__lowest_(da) == 5);
    assert(cuda::__argument::__highest_(da) == 5);
  }

  // Non-sequence values are accepted without scalar-only restrictions
  {
    auto da = cuda::__argument::__immediate{non_sequence_value{7}};
    assert(cuda::__argument::__unwrap(da).payload == 7);
  }

  // Pointer-like types can still represent a single value when explicitly wrapped that way
  {
    int value = 11;
    auto da   = cuda::__argument::__immediate{&value};
    static_assert(cuda::__argument::__traits<decltype(da)>::is_single_value);
    assert(*cuda::__argument::__unwrap(da) == 11);
  }

  // Per-segment span with runtime bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto da =
      cuda::__argument::__immediate_sequence{cuda::std::span<int>{arr, 4}, cuda::__argument::__bounds(1L, 100L)};
    assert(cuda::__argument::__unwrap(da).size() == 4);
    assert(cuda::__argument::__lowest_(da) == 1);
    assert(cuda::__argument::__highest_(da) == 100);
  }

  // Per-segment span with both bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto da    = cuda::__argument::__immediate_sequence{
      cuda::std::span<int>{arr, 4}, cuda::__argument::__bounds<1, 256>(), cuda::__argument::__bounds(10, 200)};
    static_assert(cuda::__argument::__traits<decltype(da)>::lowest == 1);
    static_assert(cuda::__argument::__traits<decltype(da)>::highest == 256);
    assert(cuda::__argument::__lowest_(da) == 10);
    assert(cuda::__argument::__highest_(da) == 200);
  }

  // Per-segment span with both bounds, runtime bounds first
  {
    int arr[4] = {10, 20, 30, 40};
    auto da    = cuda::__argument::__immediate_sequence{
      cuda::std::span<int>{arr, 4}, cuda::__argument::__bounds(10, 200), cuda::__argument::__bounds<1, 256>()};
    static_assert(cuda::__argument::__traits<decltype(da)>::lowest == 1);
    static_assert(cuda::__argument::__traits<decltype(da)>::highest == 256);
    assert(cuda::__argument::__lowest_(da) == 10);
    assert(cuda::__argument::__highest_(da) == 200);
  }

  // Per-segment via span
  {
    int arr[4] = {1, 2, 3, 4};
    auto da    = cuda::__argument::__immediate_sequence{cuda::std::span<int>{arr, 4}};
    assert(cuda::__argument::__unwrap(da).size() == 4);
    assert(cuda::__argument::__unwrap(da)[0] == 1);
    assert(cuda::__argument::__unwrap(da)[3] == 4);
  }

  // Per-segment with static bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto da =
      cuda::__argument::__immediate_sequence{cuda::std::span<int>{arr, 4}, cuda::__argument::__bounds<1, 100>()};
    assert(cuda::__argument::__unwrap(da).size() == 4);
    assert(cuda::__argument::__unwrap(da)[2] == 30);
    static_assert(cuda::__argument::__traits<decltype(da)>::lowest == 1);
    static_assert(cuda::__argument::__traits<decltype(da)>::highest == 100);
  }

  // Traits
  {
    using traits = cuda::__argument::__traits<cuda::__argument::__immediate<int>>;
    static_assert(!traits::is_deferred);
    static_assert(traits::is_single_value);
    static_assert(cuda::std::is_same_v<traits::value_type, int>);
  }

  // Sequence traits
  {
    using traits = cuda::__argument::__traits<cuda::__argument::__immediate_sequence<cuda::std::span<int>>>;
    static_assert(!traits::is_deferred);
    static_assert(!traits::is_single_value);
    static_assert(cuda::std::is_same_v<traits::value_type, cuda::std::span<int>>);
  }

  // __is_single_value_v on unwrapped types
  {
    static_assert(
      !cuda::__argument::__is_sequence_v<cuda::__argument::__traits<cuda::__argument::__immediate<int>>::value_type>);
    static_assert(
      !cuda::__argument::__traits<cuda::__argument::__immediate_sequence<cuda::std::span<int>>>::is_single_value);
  }

  // Unwrap: scalar
  {
    auto da = cuda::__argument::__immediate{7};
    auto& v = cuda::__argument::__unwrap(da);
    assert(v == 7);
    v = 8;
    assert(cuda::__argument::__unwrap(da) == 8);
  }

  // Unwrap: span
  {
    int arr[3]    = {10, 20, 30};
    auto da       = cuda::__argument::__immediate_sequence{cuda::std::span<int>{arr, 3}};
    const auto& v = cuda::__argument::__unwrap(da);
    assert(v.size() == 3);
    assert(v[1] == 20);
  }

  // Unwrap: rvalue scalar returns by value
  {
    const auto& v = cuda::__argument::__unwrap(cuda::__argument::__immediate{7});
    assert(v == 7);
  }

  // Unwrap: rvalue span returns by value
  {
    int arr[3] = {10, 20, 30};
    auto v     = cuda::__argument::__unwrap(cuda::__argument::__immediate_sequence{cuda::std::span<int>{arr, 3}});
    assert(v.size() == 3);
    assert(v[2] == 30);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
