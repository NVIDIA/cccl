//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// <ranges>

// template<class>
//  inline constexpr bool enable_borrowed_range = false;

#include <cuda/std/array>
#include <cuda/std/ranges>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif

#include "test_macros.h"

struct S
{};

__host__ __device__ void test()
{
  using cuda::std::ranges::enable_borrowed_range;
  static_assert(!enable_borrowed_range<char>);
  static_assert(!enable_borrowed_range<int>);
  static_assert(!enable_borrowed_range<double>);
  static_assert(!enable_borrowed_range<S>);

  // Sequence containers
  static_assert(!enable_borrowed_range<cuda::std::array<int, 0>>);
  static_assert(!enable_borrowed_range<cuda::std::array<int, 42>>);
#if defined(_LIBCUDACXX_HAS_VECTOR)
  static_assert(!enable_borrowed_range<cuda::std::vector<int>>);
#endif

  // Both cuda::std::span and cuda::std::string_view have their own test.
}

int main(int, char**)
{
  return 0;
}
