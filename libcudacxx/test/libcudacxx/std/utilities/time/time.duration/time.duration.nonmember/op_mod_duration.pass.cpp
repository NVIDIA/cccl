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
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator%(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "../../rep.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::chrono::nanoseconds ns1(15);
    cuda::std::chrono::nanoseconds ns2(6);
    cuda::std::chrono::nanoseconds r = ns1 % ns2;
    assert(r.count() == 3);
  }
  {
    cuda::std::chrono::microseconds us1(15);
    cuda::std::chrono::nanoseconds ns2(28);
    cuda::std::chrono::nanoseconds r = us1 % ns2;
    assert(r.count() == 20);
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s1(6);
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s2(3);
    cuda::std::chrono::duration<int, cuda::std::ratio<1, 15>> r = s1 % s2;
    assert(r.count() == 24);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  {
    cuda::std::chrono::duration<int> d(5);
    RepConstConvertibleLWG3050 x;

    {
      auto r = d % x;
      assert(r.count() == 1);
      static_assert(cuda::std::is_same_v<cuda::std::chrono::duration<long>, decltype(r)>);
    }
  }

  return 0;
}
