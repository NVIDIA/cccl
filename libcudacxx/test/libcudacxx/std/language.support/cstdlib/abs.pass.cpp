//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstdlib>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_int()
{
  ASSERT_SAME_TYPE(int, decltype(cuda::std::abs(cuda::std::declval<int>())));
  static_assert(noexcept(cuda::std::abs(cuda::std::declval<int>())), "");
  assert(cuda::std::abs(0) == 0);
  assert(cuda::std::abs(1) == 1);
  assert(cuda::std::abs(-1) == 1);
  assert(cuda::std::abs(257) == 257);
  assert(cuda::std::abs(-257) == 257);
  assert(cuda::std::abs(cuda::std::numeric_limits<int>::min() + 1) == cuda::std::numeric_limits<int>::max());
}

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_long()
{
  ASSERT_SAME_TYPE(long, decltype(cuda::std::labs(cuda::std::declval<long>())));
  static_assert(noexcept(cuda::std::abs(cuda::std::declval<long>())), "");
  assert(cuda::std::labs(0l) == 0l);
  assert(cuda::std::labs(1l) == 1l);
  assert(cuda::std::labs(-1l) == 1l);
  assert(cuda::std::labs(257l) == 257l);
  assert(cuda::std::labs(-257l) == 257l);
  assert(cuda::std::labs(cuda::std::numeric_limits<long>::min() + 1l) == cuda::std::numeric_limits<long>::max());
}

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_long_long()
{
  ASSERT_SAME_TYPE(long long, decltype(cuda::std::llabs(cuda::std::declval<long long>())));
  static_assert(noexcept(cuda::std::abs(cuda::std::declval<long long>())), "");
  assert(cuda::std::llabs(0ll) == 0ll);
  assert(cuda::std::llabs(1ll) == 1ll);
  assert(cuda::std::llabs(-1ll) == 1ll);
  assert(cuda::std::llabs(257ll) == 257ll);
  assert(cuda::std::llabs(-257ll) == 257ll);
  assert(cuda::std::llabs(cuda::std::numeric_limits<long long>::min() + 1ll)
         == cuda::std::numeric_limits<long long>::max());
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_int();
  test_long();
  test_long_long();
  return true;
}

int main(int, char**)
{
  assert(test());
#if TEST_STD_VER > 2014
  static_assert(test(), "");
#endif // TEST_STD_VER > 2014

  return 0;
}
