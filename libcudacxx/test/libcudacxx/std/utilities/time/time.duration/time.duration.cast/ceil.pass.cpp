//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// ceil

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   ceil(const duration<Rep, Period>& d);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class ToDuration, class FromDuration>
__host__ __device__ void test(const FromDuration& f, const ToDuration& d)
{
  {
    using R = decltype(cuda::std::chrono::ceil<ToDuration>(f));
    static_assert(cuda::std::is_same_v<R, ToDuration>);
    assert(cuda::std::chrono::ceil<ToDuration>(f) == d);
  }
}

int main(int, char**)
{
  //  7290000ms is 2 hours, 1 minute, and 30 seconds
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::hours(3));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::hours(-2));
  test(cuda::std::chrono::milliseconds(7290000), cuda::std::chrono::minutes(122));
  test(cuda::std::chrono::milliseconds(-7290000), cuda::std::chrono::minutes(-121));

  {
    //  9000000ms is 2 hours and 30 minutes
    constexpr cuda::std::chrono::hours h1 =
      cuda::std::chrono::ceil<cuda::std::chrono::hours>(cuda::std::chrono::milliseconds(9000000));
    static_assert(h1.count() == 3, "");
    constexpr cuda::std::chrono::hours h2 =
      cuda::std::chrono::ceil<cuda::std::chrono::hours>(cuda::std::chrono::milliseconds(-9000000));
    static_assert(h2.count() == -2, "");
  }

  return 0;
}
