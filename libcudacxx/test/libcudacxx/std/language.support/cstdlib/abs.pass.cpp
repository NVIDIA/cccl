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

#include "test_macros.h"

using I   = int;
using IL  = long;
using ILL = long long;

__host__ __device__ constexpr I abs_overload(I value)
{
  ASSERT_SAME_TYPE(I, decltype(cuda::std::abs(I{})));
  static_assert(noexcept(cuda::std::abs(I{})), "");
  return cuda::std::abs(value);
}

__host__ __device__ constexpr IL abs_overload(IL value)
{
  ASSERT_SAME_TYPE(IL, decltype(cuda::std::labs(IL{})));
  static_assert(noexcept(cuda::std::labs(IL{})), "");
  return cuda::std::labs(value);
}

__host__ __device__ constexpr ILL abs_overload(ILL value)
{
  ASSERT_SAME_TYPE(ILL, decltype(cuda::std::llabs(ILL{})));
  static_assert(noexcept(cuda::std::llabs(ILL{})), "");
  return cuda::std::llabs(value);
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_abs(T in, T ref, T zero_value)
{
  assert(abs_overload(zero_value + in) == ref);
  assert(cuda::std::abs(zero_value + in) == ref);
  return true;
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_abs(T zero_value)
{
  test_abs(T{0}, T{0}, zero_value);
  test_abs(T{1}, T{1}, zero_value);
  test_abs(T{-1}, T{1}, zero_value);
  test_abs(T{257}, T{257}, zero_value);
  test_abs(T{-257}, T{257}, zero_value);
  test_abs(cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::max(), zero_value);
  test_abs(cuda::std::numeric_limits<T>::min() + T{1}, cuda::std::numeric_limits<T>::max(), zero_value);

  static_assert(noexcept(cuda::std::abs(T{})), "");
  ASSERT_SAME_TYPE(T, decltype(cuda::std::abs(T{})));

  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test(int zero_value)
{
  test_abs(zero_value);
  test_abs(static_cast<IL>(zero_value));
  test_abs(static_cast<ILL>(zero_value));
  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
#if TEST_STD_VER >= 2014
  static_assert(test(0), "");
#endif // TEST_STD_VER >= 2014
}

int main(int, char**)
{
  volatile int zero_value = 0;
  assert(test(zero_value));
#if TEST_STD_VER >= 2014
  static_assert(test(0), "");
#endif // TEST_STD_VER >= 2014
  return 0;
}
