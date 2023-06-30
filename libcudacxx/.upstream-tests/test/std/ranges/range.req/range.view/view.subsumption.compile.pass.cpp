//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: c++20 && !nvhpc
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
  __host__ __device__ friend int* begin(View&);
  __host__ __device__ friend int* begin(View const&);
  __host__ __device__ friend int* end(View&);
  __host__ __device__ friend int* end(View const&);
};

namespace subsume_range {
  template <cuda::std::ranges::view>
  __host__ __device__ constexpr bool test() { return true; }
  template <cuda::std::ranges::range>
  __host__ __device__ constexpr bool test() { return false; }
  static_assert(test<View>(), "");
}

#ifndef __NVCOMPILER // nvbug 3885350
namespace subsume_movable {
  template <cuda::std::ranges::view>
  __host__ __device__ constexpr bool test() { return true; }
  template <cuda::std::movable>
  __host__ __device__ constexpr bool test() { return false; }
  static_assert(test<View>(), "");
}
#endif

int main(int, char**) {
  return 0;
}
