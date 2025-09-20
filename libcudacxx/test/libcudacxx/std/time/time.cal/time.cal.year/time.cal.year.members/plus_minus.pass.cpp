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

// constexpr year operator+() const noexcept;
// constexpr year operator-() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using year = cuda::std::chrono::year;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(+cuda::std::declval<year>()));
  static_assert(noexcept(-cuda::std::declval<year>()));

  static_assert(cuda::std::is_same_v<year, decltype(+cuda::std::declval<year>())>);
  static_assert(cuda::std::is_same_v<year, decltype(-cuda::std::declval<year>())>);

  for (int i = 10000; i <= 10020; ++i)
  {
    year yr(i);
    assert(static_cast<int>(+yr) == i);
    assert(static_cast<int>(-yr) == -i);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
