//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SUPPORT_FP_COMPARE_H
#define SUPPORT_FP_COMPARE_H

#include <cuda/std/algorithm> // for cuda::std::max
#include <cuda/std/cassert>
#include <cuda/std/cmath> // for cuda::std::abs

// See
// https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/boost_test/testing_tools/extended_comparison/floating_point/floating_points_comparison_theory.html

template <typename T>
__host__ __device__ bool fptest_close(T val, T expected, T eps)
{
  constexpr T zero = T(0);
  assert(eps >= zero);

  //	Handle the zero cases
  if (eps == zero)
  {
    return val == expected;
  }
  if (val == zero)
  {
    return cuda::std::abs(expected) <= eps;
  }
  if (expected == zero)
  {
    return cuda::std::abs(val) <= eps;
  }

  return cuda::std::abs(val - expected) < eps && cuda::std::abs(val - expected) / cuda::std::abs(val) < eps;
}

template <typename T>
__host__ __device__ bool fptest_close_pct(T val, T expected, T percent)
{
  constexpr T zero = T(0);
  assert(percent >= zero);

  //	Handle the zero cases
  if (percent == zero)
  {
    return val == expected;
  }
  T eps = (percent / T(100)) * cuda::std::max(cuda::std::abs(val), cuda::std::abs(expected));

  return fptest_close(val, expected, eps);
}

#endif // SUPPORT_FP_COMPARE_H
