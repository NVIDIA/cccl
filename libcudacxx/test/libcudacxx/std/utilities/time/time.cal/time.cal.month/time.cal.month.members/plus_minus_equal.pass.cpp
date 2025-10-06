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

// constexpr month& operator+=(const month& d) noexcept;
// constexpr month& operator-=(const month& d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using month  = cuda::std::chrono::month;
  using months = cuda::std::chrono::months;

  static_assert(noexcept(cuda::std::declval<month&>() += cuda::std::declval<months&>()));
  static_assert(noexcept(cuda::std::declval<month&>() -= cuda::std::declval<months&>()));
  static_assert(cuda::std::is_same_v<month&, decltype(cuda::std::declval<month&>() += cuda::std::declval<months&>())>);
  static_assert(cuda::std::is_same_v<month&, decltype(cuda::std::declval<month&>() -= cuda::std::declval<months&>())>);

  for (unsigned i = 1; i <= 10; ++i)
  {
    month m(i);
    int exp = i + 10;
    while (exp > 12)
    {
      exp -= 12;
    }
    assert(static_cast<unsigned>(m += months{10}) == static_cast<unsigned>(exp));
    assert(static_cast<unsigned>(m) == static_cast<unsigned>(exp));
    assert(m.ok());
  }

  for (unsigned i = 1; i <= 10; ++i)
  {
    month m(i);
    int exp = i - 9;
    while (exp < 1)
    {
      exp += 12;
    }
    assert(static_cast<unsigned>(m -= months{9}) == static_cast<unsigned>(exp));
    assert(static_cast<unsigned>(m) == static_cast<unsigned>(exp));
    assert(m.ok());
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
