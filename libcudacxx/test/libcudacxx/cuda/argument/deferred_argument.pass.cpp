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
  // Deferred single value via span<T, 1>
  {
    int val  = 42;
    auto def = cuda::argument::__deferred_value{cuda::std::span<int, 1>{&val, 1}};
    assert(def.arg[0] == 42);
    static_assert(cuda::argument::__traits<decltype(def)>::lowest == cuda::std::numeric_limits<int>::lowest());
    static_assert(cuda::argument::__traits<decltype(def)>::max == cuda::std::numeric_limits<int>::max());
  }

  // Deferred single value with static bounds
  {
    int val  = 42;
    auto def = cuda::argument::__deferred_value{cuda::std::span<int, 1>{&val, 1}, cuda::argument::__bounds<1, 1000>()};
    assert(def.arg[0] == 42);
    static_assert(cuda::argument::__traits<decltype(def)>::lowest == 1);
    static_assert(cuda::argument::__traits<decltype(def)>::max == 1000);
  }

  // Deferred single value via pointer (e.g., fancy iterator)
  {
    int val     = 42;
    using def_t = cuda::argument::__deferred_value<int*, cuda::argument::__static_bounds<0, 100>>;
    static_assert(cuda::argument::__traits<def_t>::lowest == 0);
    static_assert(cuda::argument::__traits<def_t>::max == 100);
    // Also verify construction works
    auto def = cuda::argument::__deferred_value{&val, cuda::argument::__bounds<0, 100>()};
    assert(def.arg == &val);
  }

  // Deferred sequence with both bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto def   = cuda::argument::__deferred_sequence{
      cuda::std::span<int>{arr, 4}, cuda::argument::__bounds<1, 4096>(), cuda::argument::__bounds(5, 100)};
    static_assert(cuda::argument::__traits<decltype(def)>::lowest == 1);
    assert(cuda::argument::__lowest_(def) == 5);
    assert(cuda::argument::__max_(def) == 100);
  }

  // Traits: deferred_value is single value
  {
    using traits = cuda::argument::__traits<cuda::argument::__deferred_value<cuda::std::span<int, 1>>>;
    static_assert(traits::is_deferred);
    static_assert(traits::is_single_value);
  }

  // Traits: deferred_value with pointer is also single value
  {
    using traits = cuda::argument::__traits<cuda::argument::__deferred_value<int*>>;
    static_assert(traits::is_deferred);
    static_assert(traits::is_single_value);
  }

  // Traits: deferred_sequence is not single value
  {
    using traits = cuda::argument::__traits<cuda::argument::__deferred_sequence<cuda::std::span<int>>>;
    static_assert(traits::is_deferred);
    static_assert(!traits::is_single_value);
  }

  // Unwrap: deferred_value
  {
    int val       = 99;
    auto def      = cuda::argument::__deferred_value{cuda::std::span<int, 1>{&val, 1}};
    const auto& v = cuda::argument::__unwrap(def);
    assert(v[0] == 99);
  }

  // Unwrap: deferred_sequence
  {
    int arr[3]    = {10, 20, 30};
    auto def      = cuda::argument::__deferred_sequence{cuda::std::span<int>{arr, 3}};
    const auto& v = cuda::argument::__unwrap(def);
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
