//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: msvc-19.16
// XFAIL: c++20 && !nvhpc && !(clang && !nvcc)
// nvbug 3885350

// <ranges>

// template<class T>
// concept view = ...;

#include <cuda/std/ranges>

#include "test_macros.h"

struct View : cuda::std::ranges::view_base {
  View() = default;
  View(View&&) = default;
  View& operator=(View&&) = default;
  TEST_HOST_DEVICE friend int* begin(View&);
  TEST_HOST_DEVICE friend int* begin(View const&);
  TEST_HOST_DEVICE friend int* end(View&);
  TEST_HOST_DEVICE friend int* end(View const&);
};

namespace subsume_range {
  template <cuda::std::ranges::view>
  TEST_HOST_DEVICE constexpr bool test() { return true; }
  template <cuda::std::ranges::range>
  TEST_HOST_DEVICE constexpr bool test() { return false; }
  static_assert(test<View>(), "");
}

#ifndef __NVCOMPILER // nvbug 3885350
namespace subsume_movable {
  template <cuda::std::ranges::view>
  TEST_HOST_DEVICE constexpr bool test() { return true; }
  template <cuda::std::movable>
  TEST_HOST_DEVICE constexpr bool test() { return false; }
  static_assert(test<View>(), "");
}
#endif

int main(int, char**) {
  return 0;
}
