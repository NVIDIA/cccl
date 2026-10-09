//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cmath>

// ilogb and logb of subnormal values return the true exponent, not the exponent of the smallest normal number.

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
TEST_FUNC _CCCL_CONSTEXPR_BIT_CAST void test_type()
{
  using limits = cuda::std::numeric_limits<T>;

  // denorm_min() is 2^(min_exponent - digits), the smallest value of the format
  constexpr int min_exp = limits::min_exponent - limits::digits;

  const T smallest = limits::denorm_min();
  assert(cuda::std::ilogb(smallest) == min_exp);
  assert(cuda::std::ilogb(-smallest) == min_exp);
  assert(cuda::std::logb(smallest) == static_cast<T>(min_exp));
  assert(cuda::std::logb(-smallest) == static_cast<T>(min_exp));

  // 3 * denorm_min() is 1.5 * 2^(min_exp + 1)
  const T three = smallest + smallest + smallest;
  assert(cuda::std::ilogb(three) == min_exp + 1);
  assert(cuda::std::logb(three) == static_cast<T>(min_exp + 1));

  // the largest subnormal is just below min(), whose exponent is min_exponent - 1
  const T largest_subnormal = limits::min() - smallest;
  assert(cuda::std::ilogb(largest_subnormal) == limits::min_exponent - 2);
  assert(cuda::std::logb(largest_subnormal) == static_cast<T>(limits::min_exponent - 2));

  // the smallest normal number is unchanged
  assert(cuda::std::ilogb(limits::min()) == limits::min_exponent - 1);
  assert(cuda::std::logb(limits::min()) == static_cast<T>(limits::min_exponent - 1));
}

TEST_FUNC _CCCL_CONSTEXPR_BIT_CAST bool test()
{
  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
  return true;
}

int main(int, char**)
{
  test();
#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  static_assert(test());
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()
  return 0;
}
