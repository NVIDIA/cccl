//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: true

// <chrono>

// duration

// template<class Rep1, class Period1, class Rep2, class Period2>
//     requires ThreeWayComparable<typename CT::rep>
//   constexpr auto operator<=>(const duration<Rep1, Period1>& lhs,
//                              const duration<Rep2, Period2>& rhs);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/ratio>

#include "test_comparisons.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::chrono::seconds s1(3);
    cuda::std::chrono::seconds s2(3);
    assert((s1 <=> s2) == cuda::std::strong_ordering::equal);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::equal));
  }
  {
    cuda::std::chrono::seconds s1(3);
    cuda::std::chrono::seconds s2(4);
    assert((s1 <=> s2) == cuda::std::strong_ordering::less);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::less));
  }
  {
    cuda::std::chrono::milliseconds s1(3);
    cuda::std::chrono::microseconds s2(3000);
    assert((s1 <=> s2) == cuda::std::strong_ordering::equal);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::equal));
  }
  {
    cuda::std::chrono::milliseconds s1(3);
    cuda::std::chrono::microseconds s2(4000);
    assert((s1 <=> s2) == cuda::std::strong_ordering::less);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::less));
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(9);
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(10);
    assert((s1 <=> s2) == cuda::std::strong_ordering::equal);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::equal));
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(10);
    cuda::std::chrono::duration<int, cuda::std::ratio<3, 5>> s2(9);
    assert((s1 <=> s2) == cuda::std::strong_ordering::greater);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::greater));
  }
  {
    cuda::std::chrono::duration<int, cuda::std::ratio<2, 3>> s1(9);
    cuda::std::chrono::duration<double, cuda::std::ratio<3, 5>> s2(10.1);
    assert((s1 <=> s2) == cuda::std::strong_ordering::less);
    assert(testOrder(s1, s2, cuda::std::strong_ordering::less));
  }

  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
