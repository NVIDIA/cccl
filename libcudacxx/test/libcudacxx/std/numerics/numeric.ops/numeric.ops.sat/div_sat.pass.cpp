//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numeric>

// template<class T>
// constexpr T div_sat(T x, T y) noexcept;                     // freestanding

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/numeric>

#include "test_macros.h"

template <class I>
__host__ __device__ constexpr bool test_div_sat(I x, I y, I res, int zero_value)
{
  assert(cuda::std::div_sat(static_cast<I>(zero_value + x), static_cast<I>(zero_value + y)) == res);
  return true;
}

template <typename I>
__host__ __device__ constexpr bool test_signed(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  ASSERT_SAME_TYPE(I, decltype(cuda::std::div_sat(I{}, I{})));
  static_assert(noexcept(cuda::std::div_sat(I{}, I{})), "");

  // Limit values (-1, 0, 1, min, max)

  test_div_sat<I>(I{-1}, I{-1}, I{1}, zero_value);
  test_div_sat<I>(I{-1}, I{1}, I{-1}, zero_value);
  test_div_sat<I>(I{-1}, minVal, I{0}, zero_value);
  test_div_sat<I>(I{-1}, maxVal, I{0}, zero_value);
  test_div_sat<I>(I{0}, I{-1}, I{0}, zero_value);
  test_div_sat<I>(I{0}, I{1}, I{0}, zero_value);
  test_div_sat<I>(I{0}, minVal, I{0}, zero_value);
  test_div_sat<I>(I{0}, maxVal, I{0}, zero_value);
  test_div_sat<I>(I{1}, I{-1}, I{-1}, zero_value);
  test_div_sat<I>(I{1}, I{1}, I{1}, zero_value);
  test_div_sat<I>(I{1}, minVal, I{0}, zero_value);
  test_div_sat<I>(I{1}, maxVal, I{0}, zero_value);
  test_div_sat<I>(minVal, I{1}, minVal, zero_value);
  test_div_sat<I>(minVal, I{-1}, maxVal, zero_value); // saturated
  test_div_sat<I>(minVal, minVal, I{1}, zero_value);
  test_div_sat<I>(minVal, maxVal, (minVal / maxVal), zero_value);
  test_div_sat<I>(maxVal, I{-1}, -maxVal, zero_value);
  test_div_sat<I>(maxVal, I{1}, maxVal, zero_value);
  test_div_sat<I>(maxVal, minVal, I{0}, zero_value);
  test_div_sat<I>(maxVal, maxVal, I{1}, zero_value);

  // No saturation (no limit values)

  test_div_sat<I>(I{27}, I{28}, I{0}, zero_value);
  test_div_sat<I>(I{28}, I{27}, I{1}, zero_value);
  {
    constexpr I lesserVal = minVal / I{2} + I{-28};
    constexpr I biggerVal = minVal / I{2} + I{-27};
    test_div_sat<I>(lesserVal, biggerVal, I{1}, zero_value);
    test_div_sat<I>(biggerVal, lesserVal, I{0}, zero_value);
  }
  {
    constexpr I lesserVal = minVal / I{2} + I{-27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_div_sat<I>(lesserVal, biggerVal, I{-1}, zero_value);
    test_div_sat<I>(biggerVal, lesserVal, I{-1}, zero_value);
  }
  {
    constexpr I lesserVal = maxVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_div_sat<I>(lesserVal, biggerVal, I{0}, zero_value);
    test_div_sat<I>(biggerVal, lesserVal, I{1}, zero_value);
  }

  return true;
}

template <class I>
__host__ __device__ constexpr bool test_unsigned(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  ASSERT_SAME_TYPE(I, decltype(cuda::std::div_sat(I{}, I{})));
  static_assert(noexcept(cuda::std::div_sat(I{}, I{})), "");

  // No limit values (0, 1, min, max)

  test_div_sat<I>(I{0}, I{1}, I{0}, zero_value);
  test_div_sat<I>(I{0}, maxVal, I{0}, zero_value);

  test_div_sat<I>(I{1}, I{1}, I{1}, zero_value);
  test_div_sat<I>(I{1}, maxVal, I{0}, zero_value);

  test_div_sat<I>(minVal, I{1}, minVal, zero_value);
  test_div_sat<I>(minVal, maxVal, I{0}, zero_value);

  test_div_sat<I>(maxVal, I{1}, maxVal, zero_value);
  test_div_sat<I>(maxVal, maxVal, I{1}, zero_value);

  // No saturation (no limit values)

  test_div_sat<I>(I{27}, I{28}, I{0}, zero_value);
  test_div_sat<I>(I{28}, I{27}, I{1}, zero_value);
  {
    constexpr I lesserVal = maxVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_div_sat<I>(lesserVal, biggerVal, I{0}, zero_value);
    test_div_sat<I>(biggerVal, lesserVal, I{1}, zero_value);
  }

  // Unsigned integer division never overflows

  return true;
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_signed<signed char>(zero_value);
  test_signed<short int>(zero_value);
  test_signed<int>(zero_value);
  test_signed<long int>(zero_value);
  test_signed<long long int>(zero_value);
#if !defined(TEST_HAS_NO_INT128_T) && !defined(TEST_COMPILER_CLANG_CUDA)
  test_signed<__int128_t>(static_cast<__int128_t>(zero_value));
#endif // !TEST_HAS_NO_INT128_T && !TEST_COMPILER_CLANG_CUDA

  test_unsigned<unsigned char>(zero_value);
  test_unsigned<unsigned short int>(zero_value);
  test_unsigned<unsigned int>(zero_value);
  test_unsigned<unsigned long int>(zero_value);
  test_unsigned<unsigned long long int>(zero_value);
#if !defined(TEST_HAS_NO_INT128_T) && !defined(TEST_COMPILER_CLANG_CUDA)
  test_unsigned<__uint128_t>(static_cast<__uint128_t>(zero_value));
#endif // !TEST_HAS_NO_INT128_T && !TEST_COMPILER_CLANG_CUDA

  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
  static_assert(test(0), "");
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
  static_assert(test(0), "");

  return 0;
}
