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

// constexpr day operator+(const day& x, const days& y) noexcept;
//   Returns: day(unsigned{x} + y.count()).
//
// constexpr day operator+(const days& x, const day& y) noexcept;
//   Returns: y + x.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using day  = cuda::std::chrono::day;
using days = cuda::std::chrono::days;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(cuda::std::declval<day>() + cuda::std::declval<days>()));
  static_assert(noexcept(cuda::std::declval<days>() + cuda::std::declval<day>()));

  static_assert(cuda::std::is_same_v<day, decltype(cuda::std::declval<day>() + cuda::std::declval<days>())>);
  static_assert(cuda::std::is_same_v<day, decltype(cuda::std::declval<days>() + cuda::std::declval<day>())>);

  day dy{12};
  for (unsigned i = 0; i <= 10; ++i)
  {
    day d1 = dy + days{i};
    day d2 = days{i} + dy;
    assert(d1 == d2);
    assert(static_cast<unsigned>(d1) == i + 12);
    assert(static_cast<unsigned>(d2) == i + 12);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
