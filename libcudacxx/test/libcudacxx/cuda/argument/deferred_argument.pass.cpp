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
#include <cuda/iterator>
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
    auto def = cuda::args::deferred{cuda::std::span<int, 1>{&val, 1}};
    assert(cuda::args::__unwrap(def)[0] == 42);
    assert(cuda::args::__access::__arg(def)[0] == 42);
    static_assert(cuda::args::__traits<decltype(def)>::lowest == cuda::std::numeric_limits<int>::lowest());
    static_assert(cuda::args::__traits<decltype(def)>::highest == (cuda::std::numeric_limits<int>::max)());
  }

  // Deferred single value with static bounds
  {
    int val  = 42;
    auto def = cuda::args::deferred{cuda::std::span<int, 1>{&val, 1}, cuda::args::bounds<1, 1000>()};
    assert(cuda::args::__unwrap(def)[0] == 42);
    static_assert(cuda::args::__traits<decltype(def)>::lowest == 1);
    static_assert(cuda::args::__traits<decltype(def)>::highest == 1000);
  }

  // Deferred single value via pointer
  {
    int val     = 42;
    using def_t = cuda::args::deferred<int*, cuda::args::static_bounds<0, 100>>;
    static_assert(cuda::args::__traits<def_t>::lowest == 0);
    static_assert(cuda::args::__traits<def_t>::highest == 100);
    // Also verify construction works
    auto def = cuda::args::deferred{&val, cuda::args::bounds<0, 100>()};
    assert(cuda::args::__unwrap(def) == &val);
  }

  // Deferred single value via fancy iterator
  {
    auto it  = cuda::counting_iterator<int>{42};
    auto def = cuda::args::deferred{it, cuda::args::bounds<0, 100>()};
    assert(cuda::args::__unwrap(def)[0] == 42);
    static_assert(cuda::args::__traits<decltype(def)>::lowest == 0);
    static_assert(cuda::args::__traits<decltype(def)>::highest == 100);
    static_assert(cuda::args::__traits<decltype(def)>::is_single_value);
  }

  // Deferred single value with both bounds, runtime bounds first
  {
    int val = 42;
    auto def =
      cuda::args::deferred{cuda::std::span<int, 1>{&val, 1}, cuda::args::bounds(5, 100), cuda::args::bounds<1, 256>()};
    static_assert(cuda::args::__traits<decltype(def)>::lowest == 1);
    static_assert(cuda::args::__traits<decltype(def)>::highest == 256);
    assert(cuda::args::__access::__runtime_bounds(def).lower() == 5);
    assert(cuda::args::__access::__runtime_bounds(def).upper() == 100);
    assert(cuda::args::__lowest_(def) == 5);
    assert(cuda::args::__highest_(def) == 100);
    cuda::args::__access::__runtime_bounds(def) = cuda::args::bounds(5, 90);
    assert(cuda::args::__highest_(def) == 90);
  }

  // Deferred sequence via fancy iterator
  {
    auto it  = cuda::counting_iterator<int>{10};
    auto def = cuda::args::deferred_sequence{it, cuda::args::bounds<0, 100>()};
    assert(cuda::args::__unwrap(def)[0] == 10);
    assert(cuda::args::__unwrap(def)[2] == 12);
    static_assert(cuda::args::__traits<decltype(def)>::lowest == 0);
    static_assert(cuda::args::__traits<decltype(def)>::highest == 100);
    static_assert(!cuda::args::__traits<decltype(def)>::is_single_value);
  }

  // Deferred sequence with both bounds
  {
    int arr[4] = {10, 20, 30, 40};
    auto def   = cuda::args::deferred_sequence{
      cuda::std::span<int>{arr, 4}, cuda::args::bounds<1, 4096>(), cuda::args::bounds(5, 100)};
    assert(cuda::args::__access::__arg(def).size() == 4);
    assert(cuda::args::__access::__runtime_bounds(def).lower() == 5);
    assert(cuda::args::__access::__runtime_bounds(def).upper() == 100);
    static_assert(cuda::args::__traits<decltype(def)>::lowest == 1);
    assert(cuda::args::__lowest_(def) == 5);
    assert(cuda::args::__highest_(def) == 100);
  }

  // Deferred sequence with both bounds, runtime bounds first
  {
    int arr[4] = {10, 20, 30, 40};
    auto def   = cuda::args::deferred_sequence{
      cuda::std::span<int>{arr, 4}, cuda::args::bounds(5, 100), cuda::args::bounds<1, 4096>()};
    static_assert(cuda::args::__traits<decltype(def)>::lowest == 1);
    static_assert(cuda::args::__traits<decltype(def)>::highest == 4096);
    assert(cuda::args::__lowest_(def) == 5);
    assert(cuda::args::__highest_(def) == 100);
  }

  // Traits: deferred is single value
  {
    using traits = cuda::args::__traits<cuda::args::deferred<cuda::std::span<int, 1>>>;
    static_assert(traits::is_deferred);
    static_assert(traits::is_single_value);
  }

  // Traits: deferred with pointer is also single value
  {
    using traits = cuda::args::__traits<cuda::args::deferred<int*>>;
    static_assert(traits::is_deferred);
    static_assert(traits::is_single_value);
  }

  // Traits: deferred_sequence is not single value
  {
    using traits = cuda::args::__traits<cuda::args::deferred_sequence<cuda::std::span<int>>>;
    static_assert(traits::is_deferred);
    static_assert(!traits::is_single_value);
  }

  // Unwrap: deferred
  {
    int val  = 99;
    auto def = cuda::args::deferred{cuda::std::span<int, 1>{&val, 1}};
    auto& v  = cuda::args::__unwrap(def);
    assert(v[0] == 99);
  }

  // Unwrap: deferred_sequence
  {
    int arr[3]    = {10, 20, 30};
    auto def      = cuda::args::deferred_sequence{cuda::std::span<int>{arr, 3}};
    const auto& v = cuda::args::__unwrap(def);
    assert(v.size() == 3);
    assert(v[1] == 20);
  }

  // Unwrap: rvalue deferred returns by value
  {
    int val = 99;
    auto v  = cuda::args::__unwrap(cuda::args::deferred{cuda::std::span<int, 1>{&val, 1}});
    assert(v[0] == 99);
  }

  // Unwrap: rvalue deferred_sequence returns by value
  {
    int arr[3] = {10, 20, 30};
    auto v     = cuda::args::__unwrap(cuda::args::deferred_sequence{cuda::std::span<int>{arr, 3}});
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
