//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month;

// constexpr month operator-(const month& x, const months& y) noexcept;
//   Returns: x + -y.
//
// constexpr months operator-(const month& x, const month& y) noexcept;
//   Returns: If x.ok() == true and y.ok() == true, returns a value m in the range
//   [months{0}, months{11}] satisfying y + m == x.
//   Otherwise the value returned is unspecified.
//   [Example: January - February == months{11}. -end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using month  = cuda::std::chrono::month;
using months = cuda::std::chrono::months;

__host__ __device__ constexpr bool test()
{
  static_assert(noexcept(cuda::std::declval<month>() - cuda::std::declval<months>()));
  static_assert(noexcept(cuda::std::declval<month>() - cuda::std::declval<month>()));

  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<month>() - cuda::std::declval<months>())>);
  static_assert(cuda::std::is_same_v<months, decltype(cuda::std::declval<month>() - cuda::std::declval<month>())>);

  month m{6};
  for (unsigned i = 1; i <= 12; ++i)
  {
    month m1 = m - months{i};
    assert(m1.ok());
    int exp = 6 - i;
    if (exp < 1)
    {
      exp += 12;
    }
    assert(static_cast<unsigned>(m1) == static_cast<unsigned>(exp));
  }

  //  Check the example
  assert(month{1} - month{2} == months{11});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
