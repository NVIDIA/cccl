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
// constexpr T sub_sat(T x, T y) noexcept;                     // freestanding

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>

template <class I>
__host__ __device__ constexpr void test_sub_sat(I x, I y, I res, int zero_value)
{
  assert(cuda::std::sub_sat(static_cast<I>(zero_value + x), static_cast<I>(zero_value + y)) == res);
}

template <class I>
__host__ __device__ constexpr void test_signed(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<I, decltype(cuda::std::sub_sat(I{}, I{}))>);
  static_assert(noexcept(cuda::std::sub_sat(I{}, I{})));

  // Limit values (-1, 0, 1, min, max)

  test_sub_sat<I>(I{-1}, I{-1}, I{0}, zero_value);
  test_sub_sat<I>(I{-1}, I{0}, I{-1}, zero_value);
  test_sub_sat<I>(I{-1}, I{1}, I{-2}, zero_value);
  test_sub_sat<I>(I{-1}, minVal, I{-1} - minVal, zero_value);
  test_sub_sat<I>(I{-1}, maxVal, I{-1} - maxVal, zero_value);

  test_sub_sat<I>(I{0}, I{-1}, I{1}, zero_value);
  test_sub_sat<I>(I{0}, I{0}, I{0}, zero_value);
  test_sub_sat<I>(I{0}, I{1}, I{-1}, zero_value);
  test_sub_sat<I>(I{0}, minVal, maxVal, zero_value); // saturated
  test_sub_sat<I>(I{0}, maxVal, -maxVal, zero_value);

  test_sub_sat<I>(minVal, I{-1}, minVal - I{-1}, zero_value);
  test_sub_sat<I>(minVal, I{0}, minVal, zero_value);
  test_sub_sat<I>(minVal, I{1}, minVal, zero_value); // saturated
  test_sub_sat<I>(minVal, minVal, I{0}, zero_value);
  test_sub_sat<I>(minVal, maxVal, minVal, zero_value); // saturated

  test_sub_sat<I>(maxVal, I{-1}, maxVal, zero_value); // saturated
  test_sub_sat<I>(maxVal, I{0}, maxVal, zero_value);
  test_sub_sat<I>(maxVal, I{1}, maxVal - I{1}, zero_value);
  test_sub_sat<I>(maxVal, minVal, maxVal, zero_value); // saturated
  test_sub_sat<I>(maxVal, maxVal, I{0}, zero_value);

  // No saturation (no limit values)

  test_sub_sat<I>(I{27}, I{-28}, I{55}, zero_value);
  test_sub_sat<I>(I{27}, I{28}, I{-1}, zero_value);
  test_sub_sat<I>(I{-27}, I{28}, I{-55}, zero_value);
  test_sub_sat<I>(I{-27}, I{-28}, I{1}, zero_value);

  // Saturation (no limit values)

  {
    constexpr I lesserVal = minVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_sub_sat<I>(lesserVal, biggerVal, minVal, zero_value); // saturated
  }
  {
    constexpr I biggerVal = maxVal / I{2} + I{28};
    constexpr I lesserVal = minVal / I{2} + I{27};
    test_sub_sat<I>(biggerVal, lesserVal, maxVal, zero_value); // saturated
  }
}

template <class I>
__host__ __device__ constexpr void test_unsigned(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<I, decltype(cuda::std::sub_sat(I{}, I{}))>);
  static_assert(noexcept(cuda::std::sub_sat(I{}, I{})));

  // Limit values (0, 1, min, max)

  test_sub_sat<I>(I{0}, I{0}, I{0}, zero_value);
  test_sub_sat<I>(I{0}, I{1}, minVal, zero_value); // saturated
  test_sub_sat<I>(I{0}, minVal, minVal, zero_value);
  test_sub_sat<I>(I{0}, maxVal, minVal, zero_value); // saturated

  test_sub_sat<I>(I{1}, I{0}, I{1}, zero_value);
  test_sub_sat<I>(I{1}, I{1}, I{0}, zero_value);
  test_sub_sat<I>(I{1}, minVal, I{1}, zero_value);
  test_sub_sat<I>(I{1}, maxVal, minVal, zero_value); // saturated

  test_sub_sat<I>(minVal, I{0}, I{0}, zero_value);
  test_sub_sat<I>(minVal, I{1}, minVal, zero_value);
  test_sub_sat<I>(minVal, maxVal, minVal, zero_value);
  test_sub_sat<I>(minVal, maxVal, minVal, zero_value);

  test_sub_sat<I>(maxVal, I{0}, maxVal, zero_value);
  test_sub_sat<I>(maxVal, I{1}, maxVal - I{1}, zero_value);
  test_sub_sat<I>(maxVal, minVal, maxVal, zero_value);
  test_sub_sat<I>(maxVal, maxVal, I{0}, zero_value);

  // Saturation (no limit values)

  {
    constexpr I lesserVal = minVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_sub_sat<I>(lesserVal, biggerVal, minVal, zero_value); // saturated
  }
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_signed<signed char>(zero_value);
  test_signed<signed short>(zero_value);
  test_signed<signed int>(zero_value);
  test_signed<signed long>(zero_value);
  test_signed<signed long long>(zero_value);
#if _CCCL_HAS_INT128()
  test_signed<__int128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  test_unsigned<unsigned char>(zero_value);
  test_unsigned<unsigned short>(zero_value);
  test_unsigned<unsigned int>(zero_value);
  test_unsigned<unsigned long>(zero_value);
  test_unsigned<unsigned long long>(zero_value);
#if _CCCL_HAS_INT128()
  test_unsigned<__uint128_t>(zero_value);
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
  static_assert(test(0));

  return 0;
}
