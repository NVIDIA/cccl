//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator==(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator!=(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::chrono::seconds s1(3);
    cuda::std::chrono::seconds s2(3);
    assert(s1 == s2);
    assert(!(s1 != s2));
  }
  {
    cuda::std::chrono::seconds s1(3);
    cuda::std::chrono::seconds s2(4);
    assert(!(s1 == s2));
    assert(s1 != s2);
  }
  {
    cuda::std::chrono::milliseconds s1(3);
    cuda::std::chrono::microseconds s2(3000);
    assert(s1 == s2);
    assert(!(s1 != s2));
  }
  {
    cuda::std::chrono::milliseconds s1(3);
    cuda::std::chrono::microseconds s2(4000);
    assert(!(s1 == s2));
    assert(s1 != s2);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(9);
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(10);
    assert(s1 == s2);
    assert(!(s1 != s2));
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(10);
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(9);
    assert(!(s1 == s2));
    assert(s1 != s2);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(9);
    cuda::std::chrono::duration<double, cuda::std::ratio<3, 5>> s2(10);
    assert(s1 == s2);
    assert(!(s1 != s2));
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
