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

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const duration<Rep1, Period>& d, const Rep2& s);

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator*(const Rep1& s, const duration<Rep2, Period>& d);

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "../../rep.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  cuda::std::chrono::nanoseconds ns(3);
  ns = ns * 5;
  assert(ns.count() == 15);
  ns = 6 * ns;
  assert(ns.count() == 90);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  { // This is related to PR#41130
    using Duration = cuda::std::chrono::nanoseconds;
    Duration d(5);
    NotARep n;
    static_assert(cuda::std::is_same_v<Duration, decltype(d * n)>);
    static_assert(cuda::std::is_same_v<Duration, decltype(n * d)>);
    d = d * n;
    assert(d.count() == 5);
    d = n * d;
    assert(d.count() == 5);
  }

  {
    cuda::std::chrono::duration<int> d(8);
    RepConstConvertibleLWG3050 x;

    {
      auto r = d * x;
      assert(r.count() == 16);
      static_assert(cuda::std::is_same_v<cuda::std::chrono::duration<long>, decltype(r)>);
    }
    {
      auto r = x * d;
      assert(r.count() == 16);
      static_assert(cuda::std::is_same_v<cuda::std::chrono::duration<long>, decltype(r)>);
    }
  }

  return 0;
}
