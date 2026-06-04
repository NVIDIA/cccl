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
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  // --- static_argument_bounds ---

  // Basic static bounds
  {
    constexpr auto b = cuda::argument::static_bounds<1, 4096>{};
    static_assert(b.lower() == 1);
    static_assert(b.upper() == 4096);
  }

  // Exact static bounds
  {
    constexpr auto b = cuda::argument::static_bounds<42, 42>{};
    static_assert(b.lower() == 42);
    static_assert(b.upper() == 42);
  }

  // Long type deduced from NTTPs
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::argument::static_bounds<0L, 1000L>::lower()), long>);
  }

#if TEST_HAS_CLASS_NTTP
  // Static bounds preserve their original NTTP types
  {
    constexpr auto b = cuda::argument::bounds<1.0f, 8.0f>();
    static_assert(b.lower() == 1.0f);
    static_assert(b.upper() == 8);
    static_assert(cuda::std::is_same_v<decltype(b.lower()), float>);
    static_assert(cuda::std::is_same_v<decltype(b.upper()), float>);
  }
#endif // TEST_HAS_CLASS_NTTP

  // --- runtime_argument_bounds ---

  // Basic runtime bounds
  {
    auto b = cuda::argument::runtime_bounds{10, 100};
    assert(b.lower() == 10);
    assert(b.upper() == 100);
    assert(b.__lower_ == 10);
    assert(b.__upper_ == 100);
    b.__upper_ = 90;
    assert(b.upper() == 90);
    static_assert(cuda::std::is_same_v<decltype(b.lower()), int>);
  }

  // --- argument_bounds factory functions ---

  // Static via factory
  {
    constexpr auto b = cuda::argument::bounds<1, 8>();
    static_assert(b.lower() == 1);
    static_assert(b.upper() == 8);
    static_assert(cuda::argument::__is_static_bounds_cv_v<decltype(b)>);
    static_assert(!cuda::argument::__is_runtime_bounds_cv_v<decltype(b)>);
    static_assert(cuda::argument::__is_bounds_v<decltype(b)>);
  }

  // Runtime via factory
  {
    auto b = cuda::argument::bounds(10, 100);
    assert(b.lower() == 10);
    assert(b.upper() == 100);
    static_assert(!cuda::argument::__is_static_bounds_cv_v<decltype(b)>);
    static_assert(cuda::argument::__is_runtime_bounds_cv_v<decltype(b)>);
    static_assert(cuda::argument::__is_bounds_v<decltype(b)>);
  }

  // Static and runtime bounds intersection
  {
    static_assert(cuda::argument::__has_bounds_intersection<int, cuda::argument::static_bounds<1, 100>>(
      cuda::argument::runtime_bounds<int>{50, 200}));
    static_assert(!cuda::argument::__has_bounds_intersection<int, cuda::argument::static_bounds<100, 200>>(
      cuda::argument::runtime_bounds<int>{0, 50}));
  }

  // Non-bounds type
  {
    static_assert(!cuda::argument::__is_bounds_v<int>);
  }

  // Bounds types accepted by argument wrapper template parameters
  {
    static_assert(cuda::argument::__valid_static_bounds_v<int, cuda::argument::no_bounds>);
    static_assert(cuda::argument::__valid_static_bounds_v<int, cuda::argument::static_bounds<1, 8>>);
    static_assert(!cuda::argument::__valid_static_bounds_v<int, cuda::argument::runtime_bounds<int>>);
    static_assert(!cuda::argument::__valid_static_bounds_v<int, int>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
