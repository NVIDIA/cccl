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

//  constexpr month& operator++() noexcept;
//  constexpr month operator++(int) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using month = cuda::std::chrono::month;

  static_assert(noexcept(++(cuda::std::declval<month&>())));
  static_assert(noexcept((cuda::std::declval<month&>())++));

  static_assert(cuda::std::is_same_v<month, decltype(cuda::std::declval<month&>()++)>);
  static_assert(cuda::std::is_same_v<month&, decltype(++cuda::std::declval<month&>())>);

  for (unsigned i = 0; i <= 15; ++i)
  {
    month m1(i);
    month m2 = m1++;
    assert(m1.ok());
    assert(m1 != m2);

    unsigned exp = i + 1;
    while (exp > 12)
    {
      exp -= 12;
    }
    assert(static_cast<unsigned>(m1) == exp);
  }
  for (unsigned i = 0; i <= 15; ++i)
  {
    month m1(i);
    month m2 = ++m1;
    assert(m1.ok());
    assert(m2.ok());
    assert(m1 == m2);

    unsigned exp = i + 1;
    while (exp > 12)
    {
      exp -= 12;
    }
    assert(static_cast<unsigned>(m1) == exp);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
