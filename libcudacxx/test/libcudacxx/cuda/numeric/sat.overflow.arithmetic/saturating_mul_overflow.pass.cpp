//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/numeric>

// template<class T>
// constexpr overflow_result<T> saturating_mul_overflow(T x, T y) noexcept;                     // freestanding

// template<class T>
// constexpr bool saturating_mul_overflow(T& result, T x, T y) noexcept;                     // freestanding

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

template <class I>
__host__ __device__ constexpr void test_saturating_mul_overflow(I x, I y, I res, bool of, int zero_value)
{
  {
    const auto [result, overflow] =
      cuda::saturating_mul_overflow(static_cast<I>(x + zero_value), static_cast<I>(y + zero_value));
    assert(result == res);
    assert(overflow == of);
  }
  {
    I result{};
    const auto overflow =
      cuda::saturating_mul_overflow(result, static_cast<I>(x + zero_value), static_cast<I>(y + zero_value));
    assert(result == res);
    assert(overflow == of);
  }
}

template <typename I>
__host__ __device__ constexpr void test_signed(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<cuda::overflow_result<I>, decltype(cuda::saturating_mul_overflow(I{}, I{}))>);
  static_assert(
    cuda::std::is_same_v<bool, decltype(cuda::saturating_mul_overflow(cuda::std::declval<I&>(), I{}, I{}))>);
  static_assert(noexcept(cuda::saturating_mul_overflow(I{}, I{})));
  static_assert(noexcept(cuda::saturating_mul_overflow(cuda::std::declval<I&>(), I{}, I{})));

  // Limit values (-1, 0, 1, min, max)

  test_saturating_mul_overflow<I>(I{-1}, I{-1}, I{1}, false, zero_value);
  test_saturating_mul_overflow<I>(I{-1}, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{-1}, I{1}, I{-1}, false, zero_value);
  test_saturating_mul_overflow<I>(I{-1}, minVal, maxVal, true, zero_value); // saturated
  test_saturating_mul_overflow<I>(I{-1}, maxVal, -maxVal, false, zero_value);

  test_saturating_mul_overflow<I>(I{0}, I{-1}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, I{1}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, minVal, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, maxVal, I{0}, false, zero_value);

  test_saturating_mul_overflow<I>(I{1}, I{-1}, I{-1}, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, I{1}, I{1}, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, minVal, minVal, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, maxVal, maxVal, false, zero_value);

  test_saturating_mul_overflow<I>(minVal, I{-1}, maxVal, true, zero_value); // saturated
  test_saturating_mul_overflow<I>(minVal, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(minVal, I{1}, minVal, false, zero_value);
  test_saturating_mul_overflow<I>(minVal, minVal, maxVal, true, zero_value); // saturated
  test_saturating_mul_overflow<I>(minVal, maxVal, minVal, true, zero_value); // saturated

  test_saturating_mul_overflow<I>(maxVal, I{-1}, -maxVal, false, zero_value);
  test_saturating_mul_overflow<I>(maxVal, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(maxVal, I{1}, maxVal, false, zero_value); // saturated
  test_saturating_mul_overflow<I>(maxVal, minVal, minVal, true, zero_value); // saturated

  // No saturation (no limit values)

  test_saturating_mul_overflow<I>(I{27}, I{2}, I{54}, false, zero_value);
  test_saturating_mul_overflow<I>(I{2}, I{28}, I{56}, false, zero_value);

  // Saturation (no limit values)

  {
    constexpr I x = minVal / I{2} + I{27};
    constexpr I y = minVal / I{2} + I{28};
    test_saturating_mul_overflow<I>(x, y, maxVal, true, zero_value); // saturated
  }
  {
    constexpr I x = minVal / I{2} + I{27};
    constexpr I y = maxVal / I{2} + I{28};
    test_saturating_mul_overflow<I>(x, y, minVal, true, zero_value); // saturated
  }
  {
    constexpr I x = maxVal / I{2} + I{27};
    constexpr I y = minVal / I{2} + I{28};
    test_saturating_mul_overflow<I>(x, y, minVal, true, zero_value); // saturated
  }
  {
    constexpr I x = maxVal / I{2} + I{27};
    constexpr I y = maxVal / I{2} + I{28};
    test_saturating_mul_overflow<I>(x, y, maxVal, true, zero_value); // saturated
  }
}

template <class I>
__host__ __device__ constexpr void test_unsigned(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<cuda::overflow_result<I>, decltype(cuda::saturating_mul_overflow(I{}, I{}))>);
  static_assert(
    cuda::std::is_same_v<bool, decltype(cuda::saturating_mul_overflow(cuda::std::declval<I&>(), I{}, I{}))>);
  static_assert(noexcept(cuda::saturating_mul_overflow(I{}, I{})));
  static_assert(noexcept(cuda::saturating_mul_overflow(cuda::std::declval<I&>(), I{}, I{})));

  // No saturation (0, 1)

  test_saturating_mul_overflow<I>(I{0}, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, I{1}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, minVal, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{0}, maxVal, I{0}, false, zero_value);

  test_saturating_mul_overflow<I>(I{1}, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, I{1}, I{1}, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, minVal, minVal, false, zero_value);
  test_saturating_mul_overflow<I>(I{1}, maxVal, maxVal, false, zero_value);

  test_saturating_mul_overflow<I>(minVal, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(minVal, I{1}, minVal, false, zero_value);
  test_saturating_mul_overflow<I>(minVal, maxVal, minVal, false, zero_value);
  test_saturating_mul_overflow<I>(minVal, maxVal, minVal, false, zero_value);

  test_saturating_mul_overflow<I>(maxVal, I{0}, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(maxVal, I{1}, maxVal, false, zero_value);
  test_saturating_mul_overflow<I>(maxVal, minVal, I{0}, false, zero_value);
  test_saturating_mul_overflow<I>(maxVal, maxVal, maxVal, true, zero_value); // saturated

  // No saturation (no limit values)

  test_saturating_mul_overflow<I>(I{28}, I{2}, I{56}, false, zero_value);

  // Saturation (no limit values

  {
    constexpr I x = maxVal / I{2} + I{27};
    constexpr I y = maxVal / I{2} + I{28};
    test_saturating_mul_overflow<I>(x, y, maxVal, true, zero_value); // saturated
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
