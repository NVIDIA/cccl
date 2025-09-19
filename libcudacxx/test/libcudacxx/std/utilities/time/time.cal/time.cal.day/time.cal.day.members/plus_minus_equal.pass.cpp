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

// constexpr day& operator+=(const days& d) noexcept;
// constexpr day& operator-=(const days& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using day  = cuda::std::chrono::day;
using days = cuda::std::chrono::days;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(cuda::std::declval<day&>() += cuda::std::declval<days>()));
  static_assert(noexcept(cuda::std::declval<day&>() -= cuda::std::declval<days>()));

  static_assert(cuda::std::is_same_v<day&, decltype(cuda::std::declval<day&>() += cuda::std::declval<days>())>);
  static_assert(cuda::std::is_same_v<day&, decltype(cuda::std::declval<day&>() -= cuda::std::declval<days>())>);

  for (unsigned i = 0; i <= 10; ++i)
  {
    day d(i);
    assert(static_cast<unsigned>(d += days{22}) == i + 22);
    assert(static_cast<unsigned>(d) == i + 22);
    assert(static_cast<unsigned>(d -= days{12}) == i + 10);
    assert(static_cast<unsigned>(d) == i + 10);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
