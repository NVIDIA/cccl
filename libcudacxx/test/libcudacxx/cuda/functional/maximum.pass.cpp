//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>

#include "min_max_common.h"
#include "test_macros.h"

template <typename T, T lhs, T rhs, T expected>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  test_op<cuda::maximum<T>, T, lhs, rhs, expected>();
  test_op<cuda::maximum<>, T, lhs, rhs, expected>();
  test_op<cuda::maximum<void>, T, lhs, rhs, expected>();
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<int, 0, 1, 1>();
  test<int, 1, 0, 1>();
  test<int, 0, 0, 0>();
  test<int, -1, 1, 1>();
  test<char, 'a', 'b', 'b'>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014
  return 0;
}
