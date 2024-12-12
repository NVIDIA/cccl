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

#if _CCCL_HAS_CONSTEXPR_INT_ABS && TEST_STD_VER >= 2014
#  define ALLOW_CONSTEXPR 1
#  define CONSTEXPR       constexpr
#else
#  define CONSTEXPR
#endif

__host__ __device__ CONSTEXPR int abs_overload(int value)
{
  ASSERT_SAME_TYPE(int, decltype(cuda::std::abs(cuda::std::declval<int>())));
  static_assert(noexcept(cuda::std::abs(cuda::std::declval<int>())), "");
  return cuda::std::abs(value);
}

__host__ __device__ CONSTEXPR long abs_overload(long value)
{
  ASSERT_SAME_TYPE(long, decltype(cuda::std::labs(cuda::std::declval<long>())));
  static_assert(noexcept(cuda::std::labs(cuda::std::declval<long>())), "");
  return cuda::std::labs(value);
}

__host__ __device__ CONSTEXPR long long abs_overload(long long value)
{
  ASSERT_SAME_TYPE(long long, decltype(cuda::std::llabs(cuda::std::declval<long long>())));
  static_assert(noexcept(cuda::std::llabs(cuda::std::declval<long long>())), "");
  return cuda::std::llabs(value);
}

template <class T>
__host__ __device__ CONSTEXPR bool test_abs(T zero_value)
{
  assert(abs_overload(zero_value + T{0}) == T{0});
  assert(abs_overload(zero_value + T{1}) == T{1});
  assert(abs_overload(zero_value + T{-1}) == T{1});
  assert(abs_overload(zero_value + T{257}) == T{257});
  assert(abs_overload(zero_value + T{-257}) == T{257});
  assert(abs_overload(zero_value + cuda::std::numeric_limits<T>::max()) == cuda::std::numeric_limits<T>::max());
  assert(abs_overload(zero_value + cuda::std::numeric_limits<T>::min() + T{1}) == cuda::std::numeric_limits<T>::max());
  return true;
}

__host__ __device__ CONSTEXPR bool test(int zero_value)
{
  test_abs(zero_value);
  test_abs(static_cast<long>(zero_value));
  test_abs(static_cast<long long>(zero_value));
  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
}

int main(int, char**)
{
  volatile int zero_value = 0;
  assert(test(zero_value));
#if ALLOW_CONSTEXPR
  static_assert(test(0), "");
#endif // _CCCL_HAS_CONSTEXPR_INT_ABS
  return 0;
}
