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
#include <cuda/std/array>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

enum class color
{
  red,
  green,
  blue
};

TEST_FUNC void test()
{
  // --- __is_single_value_v ---

  static_assert(cuda::argument::__is_single_value_v<int>);
  static_assert(cuda::argument::__is_single_value_v<float>);
  static_assert(cuda::argument::__is_single_value_v<double>);
  static_assert(cuda::argument::__is_single_value_v<const int>);
  static_assert(cuda::argument::__is_single_value_v<color>);
  static_assert(cuda::argument::__is_single_value_v<cuda::std::span<int, 1>>);
  static_assert(!cuda::argument::__is_single_value_v<int*>);
  static_assert(!cuda::argument::__is_single_value_v<cuda::std::span<int>>);
  static_assert(!cuda::argument::__is_single_value_v<cuda::std::span<int, 4>>);
  static_assert(!cuda::argument::__is_single_value_v<cuda::std::array<int, 3>>);

  // --- argument_traits: is_deferred ---

  static_assert(!cuda::argument::__traits<int>::is_deferred);
  static_assert(!cuda::argument::__traits<cuda::argument::__immediate<int>>::is_deferred);
  static_assert(!cuda::argument::__traits<cuda::argument::__constant<42>>::is_deferred);
  static_assert(cuda::argument::__traits<cuda::argument::__deferred_value<cuda::std::span<int, 1>>>::is_deferred);
  static_assert(cuda::argument::__traits<cuda::argument::__deferred_sequence<cuda::std::span<int>>>::is_deferred);

  // --- argument_traits: is_single_value ---

  static_assert(cuda::argument::__traits<int>::is_single_value);
  static_assert(!cuda::argument::__traits<int*>::is_single_value);
  static_assert(cuda::argument::__traits<cuda::argument::__immediate<int>>::is_single_value);
  static_assert(!cuda::argument::__traits<cuda::argument::__immediate<cuda::std::span<int>>>::is_single_value);
  static_assert(cuda::argument::__traits<cuda::argument::__constant<42>>::is_single_value);
  static_assert(cuda::argument::__traits<cuda::argument::__deferred_value<int*>>::is_single_value);
  static_assert(!cuda::argument::__traits<cuda::argument::__deferred_sequence<cuda::std::span<int>>>::is_single_value);

  // --- argument_traits: value_type ---

  static_assert(cuda::std::is_same_v<cuda::argument::__traits<int>::value_type, int>);
  static_assert(cuda::std::is_same_v<cuda::argument::__traits<cuda::argument::__immediate<int>>::value_type, int>);
  static_assert(cuda::std::is_same_v<cuda::argument::__traits<cuda::argument::__constant<42>>::value_type, int>);

  // --- argument_traits: lowest / max ---

  static_assert(cuda::argument::__traits<int>::lowest == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::argument::__traits<int>::max == cuda::std::numeric_limits<int>::max());
  static_assert(cuda::argument::__traits<const int>::lowest == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::argument::__traits<int&>::max == cuda::std::numeric_limits<int>::max());
  static_assert(cuda::argument::__traits<float>::lowest == cuda::std::numeric_limits<float>::lowest());
  static_assert(cuda::argument::__traits<float>::max == cuda::std::numeric_limits<float>::max());
  static_assert(
    cuda::argument::__traits<const cuda::argument::__immediate<int, cuda::argument::__static_bounds<1, 8>>>::lowest
    == 1);
  static_assert(
    cuda::argument::__traits<cuda::argument::__immediate<int, cuda::argument::__static_bounds<1, 8>>&>::max == 8);

  // --- Free function bounds on plain values ---

  static_assert(cuda::argument::__lowest_(42) == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::argument::__max_(42) == cuda::std::numeric_limits<int>::max());
  static_assert(cuda::argument::__lowest_(1.0f) == cuda::std::numeric_limits<float>::lowest());
  static_assert(cuda::argument::__max_(1.0f) == cuda::std::numeric_limits<float>::max());

#if TEST_HAS_CLASS_NTTP
  // --- is_single_value with array constant ---
  using arr_t = cuda::std::array<int, 3>;
  static_assert(!cuda::argument::__traits<cuda::argument::__constant<arr_t{1, 2, 3}>>::is_single_value);
#endif // TEST_HAS_CLASS_NTTP
}

int main(int, char**)
{
  test();
  return 0;
}
