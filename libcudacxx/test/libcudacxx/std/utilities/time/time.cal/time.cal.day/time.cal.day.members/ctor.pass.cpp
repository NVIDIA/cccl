//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class day;

//                     day() = default;
//  explicit constexpr day(unsigned d) noexcept;
//  explicit constexpr operator unsigned() const noexcept;

//  Effects: Constructs an object of type day by initializing d_ with d.
//    The value held is unspecified if d is not in the range [0, 255].

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using day = cuda::std::chrono::day;

  static_assert(noexcept(day{}));
  static_assert(noexcept(day(0U)));
  static_assert(noexcept(static_cast<unsigned>(day(0U))));

  {
    day d{};
    assert(static_cast<unsigned>(d) == 0);
  }

  {
    day d{1};
    assert(static_cast<unsigned>(d) == 1);
  }

  for (unsigned i = 0; i <= 255; ++i)
  {
    day d(i);
    assert(static_cast<unsigned>(d) == i);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
