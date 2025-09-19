//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// abs

// template <class Rep, class Period>
//   constexpr duration<Rep, Period> abs(duration<Rep, Period> d)

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class Duration>
__host__ __device__ void test(const Duration& f, const Duration& d)
{
  {
    using R = decltype(cuda::std::chrono::abs(f));
    static_assert(cuda::std::is_same_v<R, Duration>);
    assert(cuda::std::chrono::abs(f) == d);
  }
}

int main(int, char**)
{
  //  7290000ms is 2 hours, 1 minute, and 30 seconds
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::milliseconds(7290000));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::milliseconds(7290000));
  test(cuda::std::chrono::minutes(122), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::minutes(-122), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::hours(0), cuda::std::chrono::hours(0));

  {
    //  9000000ms is 2 hours and 30 minutes
    constexpr cuda::std::chrono::hours h1 = cuda::std::chrono::abs(cuda::std::chrono::hours(-3));
    static_assert(h1.count() == 3, "");
    constexpr cuda::std::chrono::hours h2 = cuda::std::chrono::abs(cuda::std::chrono::hours(3));
    static_assert(h2.count() == 3, "");
  }

  {
    //  Make sure it works for durations that are not LCD'ed - example from LWG3091
    constexpr auto d = cuda::std::chrono::abs(cuda::std::chrono::duration<int, cuda::std::ratio<60, 100>>{2});
    static_assert(d.count() == 2, "");
  }

  return 0;
}
