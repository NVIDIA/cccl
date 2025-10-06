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

// constexpr year& operator+=(const years& d) noexcept;
// constexpr year& operator-=(const years& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using year  = cuda::std::chrono::year;
using years = cuda::std::chrono::years;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(cuda::std::declval<year&>() += cuda::std::declval<years>()));
  static_assert(noexcept(cuda::std::declval<year&>() -= cuda::std::declval<years>()));

  static_assert(cuda::std::is_same_v<year&, decltype(cuda::std::declval<year&>() += cuda::std::declval<years>())>);
  static_assert(cuda::std::is_same_v<year&, decltype(cuda::std::declval<year&>() -= cuda::std::declval<years>())>);

  for (int i = 10000; i <= 10020; ++i)
  {
    year yr(i);
    assert(static_cast<int>(yr += years{10}) == i + 10);
    assert(static_cast<int>(yr) == i + 10);
    assert(static_cast<int>(yr -= years{9}) == i + 1);
    assert(static_cast<int>(yr) == i + 1);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
