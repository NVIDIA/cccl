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
__host__ __device__ constexpr bool do_test()
{
  return test_op<cuda::minimum<T>, T, lhs, rhs, expected>() && test_op<cuda::minimum<>, T, lhs, rhs, expected>()
      && test_op<cuda::minimum<void>, T, lhs, rhs, expected>();
}

__host__ __device__ constexpr bool test()
{
  return do_test<int, 0, 1, 0>() && do_test<int, 1, 0, 0>() && do_test<int, 0, 0, 0>() && do_test<int, -1, 1, -1>()
      && do_test<char, 'a', 'b', 'a'>();
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2017
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test());
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif
  return 0;
}
