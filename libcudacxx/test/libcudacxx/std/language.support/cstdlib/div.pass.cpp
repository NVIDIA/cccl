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

_CCCL_DIAG_SUPPRESS_MSVC(4244) // conversion from fp to integral, possible loss of data

using I   = int;
using IL  = long;
using ILL = long long;

__host__ __device__ TEST_CONSTEXPR_CXX14 cuda::std::div_t div_overload(I x, I y)
{
  ASSERT_SAME_TYPE(cuda::std::div_t, decltype(cuda::std::div(I{}, I{})));
  static_assert(noexcept(cuda::std::div(I{}, I{})), "");
  return cuda::std::div(x, y);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 cuda::std::ldiv_t div_overload(IL x, IL y)
{
  ASSERT_SAME_TYPE(cuda::std::ldiv_t, decltype(cuda::std::ldiv(IL{}, IL{})));
  static_assert(noexcept(cuda::std::ldiv(IL{}, IL{})), "");
  return cuda::std::ldiv(x, y);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 cuda::std::lldiv_t div_overload(ILL x, ILL y)
{
  ASSERT_SAME_TYPE(cuda::std::lldiv_t, decltype(cuda::std::lldiv(ILL{}, ILL{})));
  static_assert(noexcept(cuda::std::lldiv(ILL{}, ILL{})), "");
  return cuda::std::lldiv(x, y);
}

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_div(T x_in, T y_in, T x_ref, T r_ref, T zero_value)
{
  auto result = div_overload(zero_value + x_in, zero_value + y_in);
  assert(result.quot == x_ref && result.rem == r_ref);

  result = cuda::std::div(zero_value + x_in, zero_value + y_in);
  assert(result.quot == x_ref && result.rem == r_ref);
}

template <class T, class Ret>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test_div(T zero_value)
{
  test_div(T{0}, T{1}, T{0}, T{0}, zero_value);
  test_div(T{0}, cuda::std::numeric_limits<T>::max(), T{0}, T{0}, zero_value);
  test_div(T{0}, cuda::std::numeric_limits<T>::min(), T{0}, T{0}, zero_value);

  test_div(T{1}, T{1}, T{1}, T{0}, zero_value);
  test_div(T{1}, T{-1}, T{-1}, T{0}, zero_value);
  test_div(T{-1}, T{1}, T{-1}, T{0}, zero_value);
  test_div(T{-1}, T{-1}, T{1}, T{0}, zero_value);

  test_div(T{20}, T{3}, T{6}, T{2}, zero_value);
  test_div(T{20}, T{-3}, T{-6}, T{2}, zero_value);
  test_div(T{-20}, T{3}, T{-6}, T{-2}, zero_value);
  test_div(T{-20}, T{-3}, T{6}, T{-2}, zero_value);

  static_assert(noexcept(cuda::std::div(T{}, T{})), "");
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(T{}, T{})));
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(T{}, float{})));
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(float{}, T{})));
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(T{}, double{})));
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(double{}, T{})));
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(T{}, unsigned{})));
  ASSERT_SAME_TYPE(Ret, decltype(cuda::std::div(unsigned{}, T{})));

  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test(int zero_value)
{
  test_div<I, cuda::std::div_t>(zero_value);
  test_div<IL, cuda::std::ldiv_t>(static_cast<IL>(zero_value));
  test_div<ILL, cuda::std::lldiv_t>(static_cast<ILL>(zero_value));

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
