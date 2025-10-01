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
//   operator%(const duration<Rep1, Period>& d, const Rep2& s)

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "../../rep.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using Dur = cuda::std::chrono::nanoseconds;
  Dur ns(15);
  static_assert(cuda::std::is_same_v<Dur, decltype(ns % 6)>);
  ns = ns % 6;
  assert(ns.count() == 3);
  return true;
}
int main(int, char**)
{
  test();
  static_assert(test());

  { // This is PR#41130
    using Duration = cuda::std::chrono::seconds;
    Duration d(5);
    NotARep n;
    static_assert(cuda::std::is_same_v<Duration, decltype(d % n)>);
    d = d % n;
    assert(d.count() == 5);
  }

  return 0;
}
