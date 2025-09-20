//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

//                     year() = default;
//  explicit constexpr year(int m) noexcept;
//  explicit constexpr operator int() const noexcept;

//  Effects: Constructs an object of type year by initializing y_ with y.
//    The value held is unspecified if d is not in the range [0, 255].

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using year = cuda::std::chrono::year;

  static_assert(noexcept(year{}));
  static_assert(noexcept(year(0U)));
  static_assert(noexcept(static_cast<int>(year(0U))));

  {
    year y{};
    assert(static_cast<int>(y) == 0);
  }

  {
    year y{1};
    assert(static_cast<int>(y) == 1);
  }

  for (int i = 0; i <= 2550; i += 7)
  {
    year yr(i);
    assert(static_cast<int>(yr) == i);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
