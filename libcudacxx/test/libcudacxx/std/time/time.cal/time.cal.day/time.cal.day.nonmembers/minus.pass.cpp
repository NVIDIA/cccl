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

// constexpr day operator-(const day& x, const days& y) noexcept;
//   Returns: x + -y.
//
// constexpr days operator-(const day& x, const day& y) noexcept;
//   Returns: days{int(unsigned{x}) - int(unsigned{y}).

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using day  = cuda::std::chrono::day;
using days = cuda::std::chrono::days;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(cuda::std::declval<day>() - cuda::std::declval<days>()));
  static_assert(noexcept(cuda::std::declval<day>() - cuda::std::declval<day>()));

  static_assert(cuda::std::is_same_v<day, decltype(cuda::std::declval<day>() - cuda::std::declval<days>())>);
  static_assert(cuda::std::is_same_v<days, decltype(cuda::std::declval<day>() - cuda::std::declval<day>())>);

  day dy{12};
  for (unsigned i = 0; i <= 10; ++i)
  {
    day d1   = dy - days{i};
    days off = dy - day{i};
    assert(static_cast<unsigned>(d1) == 12 - i);
    assert(off.count() == static_cast<int>(12 - i)); // days is signed
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
