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
// constexpr T add_sat(T x, T y) noexcept;                     // freestanding

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>

template <class I>
__host__ __device__ constexpr void test_add_sat(I x, I y, I res, int zero_value)
{
  if constexpr (cuda::std::is_same_v<I, uint16_t>)
  {
    auto ret = cuda::std::add_sat(static_cast<I>(zero_value + x), static_cast<I>(zero_value + y));
    printf("TEST: %d   %d  (%d)\n", (int) ret, (int) res, zero_value);
  }
  assert(cuda::std::add_sat(static_cast<I>(zero_value + x), static_cast<I>(zero_value + y)) == res);
}

template <class I>
__host__ __device__ constexpr void test_signed(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<I, decltype(cuda::std::add_sat(I{}, I{}))>);
  static_assert(noexcept(cuda::std::add_sat(I{}, I{})));

  // Limit values (-1, 0, 1, min, max)

  test_add_sat<I>(I{-1}, I{-1}, I{-2}, zero_value);
  test_add_sat<I>(I{-1}, I{0}, I{-1}, zero_value);
  test_add_sat<I>(I{-1}, I{1}, I{0}, zero_value);
  test_add_sat<I>(I{-1}, minVal, minVal, zero_value); // saturated
  test_add_sat<I>(I{-1}, maxVal, I{-1} + maxVal, zero_value);

  test_add_sat<I>(I{0}, I{-1}, I{-1}, zero_value);
  test_add_sat<I>(I{0}, I{0}, I{0}, zero_value);
  test_add_sat<I>(I{0}, I{1}, I{1}, zero_value);
  test_add_sat<I>(I{0}, minVal, minVal, zero_value);
  test_add_sat<I>(I{0}, maxVal, maxVal, zero_value);

  test_add_sat<I>(I{1}, I{-1}, I{0}, zero_value);
  test_add_sat<I>(I{1}, I{0}, I{1}, zero_value);
  test_add_sat<I>(I{1}, I{1}, I{2}, zero_value);
  test_add_sat<I>(I{1}, minVal, I{1} + minVal, zero_value);
  test_add_sat<I>(I{1}, maxVal, maxVal, zero_value); // saturated

  test_add_sat<I>(minVal, I{-1}, minVal, zero_value); // saturated
  test_add_sat<I>(minVal, I{0}, minVal, zero_value);
  test_add_sat<I>(minVal, I{1}, minVal + I{1}, zero_value);
  test_add_sat<I>(minVal, minVal, minVal, zero_value); // saturated
  test_add_sat<I>(minVal, maxVal, I{-1}, zero_value);

  test_add_sat<I>(maxVal, I{-1}, maxVal + I{-1}, zero_value);
  test_add_sat<I>(maxVal, I{0}, maxVal, zero_value);
  test_add_sat<I>(maxVal, I{1}, maxVal, zero_value); // saturated
  test_add_sat<I>(maxVal, minVal, I{-1}, zero_value);
  test_add_sat<I>(maxVal, maxVal, maxVal, zero_value); // saturated

  // No saturation (no limit values)

  test_add_sat<I>(I{-27}, I{28}, I{1}, zero_value);
  test_add_sat<I>(I{27}, I{28}, I{55}, zero_value);
  {
    constexpr I x = maxVal / I{2} + I{27};
    constexpr I y = maxVal / I{2} + I{28};
    test_add_sat<I>(x, y, maxVal, zero_value);
  }

  // Saturation (no limit values)

  {
    constexpr I x = minVal / I{2} + I{-27};
    constexpr I y = minVal / I{2} + I{-28};
    test_add_sat<I>(x, y, minVal, zero_value); // saturated
  }
  {
    constexpr I x = maxVal / I{2} + I{27};
    constexpr I y = maxVal / I{2} + I{28};
    test_add_sat<I>(x, y, maxVal, zero_value); // saturated
  }
}

template <class I>
__host__ __device__ constexpr void test_unsigned(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<I, decltype(cuda::std::add_sat(I{}, I{}))>);
  static_assert(noexcept(cuda::std::add_sat(I{}, I{})), "");

  // Litmit values (0, 1, min, max)
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{0}, I{0}, I{0}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{0}, I{1}, I{1}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{0}, minVal, I{0}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{0}, maxVal, maxVal, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{1}, I{0}, I{1}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{1}, I{1}, I{2}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{1}, minVal, I{1}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(I{1}, maxVal, maxVal, zero_value);
  printf("%d\n", __LINE__); // saturated
  test_add_sat<I>(minVal, I{0}, I{0}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(minVal, I{1}, I{1}, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(minVal, minVal, minVal, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(minVal, maxVal, maxVal, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(maxVal, I{0}, maxVal, zero_value);
  test_add_sat<I>(maxVal, I{1}, maxVal, zero_value);
  printf("%d\n", __LINE__); // saturated
  test_add_sat<I>(maxVal, minVal, maxVal, zero_value);
  printf("%d\n", __LINE__);
  test_add_sat<I>(maxVal, maxVal, maxVal, zero_value);
  printf("%d\n", __LINE__); // saturated

  // No saturation (no limit values)

  test_add_sat<I>(I{27}, I{28}, I{55}, zero_value);

  // Saturation (no limit values)

  {
    constexpr I x = maxVal / I{2} + I{27};
    constexpr I y = maxVal / I{2} + I{28};
    printf("%d\n", __LINE__); // saturated
    test_add_sat<I>(x, y, maxVal, zero_value); // saturated
    printf("%d\n", __LINE__); // saturated
    test_add_sat<I>(x, maxVal, maxVal, zero_value); // saturated
    printf("%d\n", __LINE__); // saturated
    test_add_sat<I>(maxVal, y, maxVal, zero_value); // saturated
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
  // static_assert(test(0));

  return 0;
}
