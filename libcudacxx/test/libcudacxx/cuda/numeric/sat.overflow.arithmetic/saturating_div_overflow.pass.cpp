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
// constexpr overflow_result<T> saturating_div_overflow(T x, T y) noexcept;                     // freestanding

// template<class T>
// constexpr bool saturating_div_overflow(T& result, T x, T y) noexcept;                     // freestanding

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

template <class I>
__host__ __device__ constexpr void test_saturating_div_overflow(I x, I y, I res, bool of, int zero_value)
{
  {
    const auto [result, overflow] =
      cuda::saturating_div_overflow(static_cast<I>(x + zero_value), static_cast<I>(y + zero_value));
    assert(result == res);
    assert(overflow == of);
  }
  {
    I result{};
    const auto overflow =
      cuda::saturating_div_overflow(result, static_cast<I>(x + zero_value), static_cast<I>(y + zero_value));
    assert(result == res);
    assert(overflow == of);
  }
}

template <typename I>
__host__ __device__ constexpr void test_signed(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<cuda::overflow_result<I>, decltype(cuda::saturating_div_overflow(I{}, I{}))>);
  static_assert(
    cuda::std::is_same_v<bool, decltype(cuda::saturating_div_overflow(cuda::std::declval<I&>(), I{}, I{}))>);
  static_assert(noexcept(cuda::saturating_div_overflow(I{}, I{})));
  static_assert(noexcept(cuda::saturating_div_overflow(cuda::std::declval<I&>(), I{}, I{})));

  // Limit values (-1, 0, 1, min, max)

  test_saturating_div_overflow<I>(I{-1}, I{-1}, I{1}, false, zero_value);
  test_saturating_div_overflow<I>(I{-1}, I{1}, I{-1}, false, zero_value);
  test_saturating_div_overflow<I>(I{-1}, minVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{-1}, maxVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{0}, I{-1}, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{0}, I{1}, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{0}, minVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{0}, maxVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{1}, I{-1}, I{-1}, false, zero_value);
  test_saturating_div_overflow<I>(I{1}, I{1}, I{1}, false, zero_value);
  test_saturating_div_overflow<I>(I{1}, minVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{1}, maxVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(minVal, I{1}, minVal, false, zero_value);
  test_saturating_div_overflow<I>(minVal, I{-1}, maxVal, true, zero_value); // saturated
  test_saturating_div_overflow<I>(minVal, minVal, I{1}, false, zero_value);
  test_saturating_div_overflow<I>(minVal, maxVal, (minVal / maxVal), false, zero_value);
  test_saturating_div_overflow<I>(maxVal, I{-1}, -maxVal, false, zero_value);
  test_saturating_div_overflow<I>(maxVal, I{1}, maxVal, false, zero_value);
  test_saturating_div_overflow<I>(maxVal, minVal, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(maxVal, maxVal, I{1}, false, zero_value);

  // No saturation (no limit values)

  test_saturating_div_overflow<I>(I{27}, I{28}, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{28}, I{27}, I{1}, false, zero_value);
  {
    constexpr I lesserVal = minVal / I{2} + I{-28};
    constexpr I biggerVal = minVal / I{2} + I{-27};
    test_saturating_div_overflow<I>(lesserVal, biggerVal, I{1}, false, zero_value);
    test_saturating_div_overflow<I>(biggerVal, lesserVal, I{0}, false, zero_value);
  }
  {
    constexpr I lesserVal = minVal / I{2} + I{-27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_saturating_div_overflow<I>(lesserVal, biggerVal, I{-1}, false, zero_value);
    test_saturating_div_overflow<I>(biggerVal, lesserVal, I{-1}, false, zero_value);
  }
  {
    constexpr I lesserVal = maxVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_saturating_div_overflow<I>(lesserVal, biggerVal, I{0}, false, zero_value);
    test_saturating_div_overflow<I>(biggerVal, lesserVal, I{1}, false, zero_value);
  }
}

template <class I>
__host__ __device__ constexpr void test_unsigned(int zero_value)
{
  constexpr auto minVal = cuda::std::numeric_limits<I>::min();
  constexpr auto maxVal = cuda::std::numeric_limits<I>::max();

  static_assert(cuda::std::is_same_v<cuda::overflow_result<I>, decltype(cuda::saturating_div_overflow(I{}, I{}))>);
  static_assert(
    cuda::std::is_same_v<bool, decltype(cuda::saturating_div_overflow(cuda::std::declval<I&>(), I{}, I{}))>);
  static_assert(noexcept(cuda::saturating_div_overflow(I{}, I{})));
  static_assert(noexcept(cuda::saturating_div_overflow(cuda::std::declval<I&>(), I{}, I{})));

  // No limit values (0, 1, min, max)

  test_saturating_div_overflow<I>(I{0}, I{1}, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{0}, maxVal, I{0}, false, zero_value);

  test_saturating_div_overflow<I>(I{1}, I{1}, I{1}, false, zero_value);
  test_saturating_div_overflow<I>(I{1}, maxVal, I{0}, false, zero_value);

  test_saturating_div_overflow<I>(minVal, I{1}, minVal, false, zero_value);
  test_saturating_div_overflow<I>(minVal, maxVal, I{0}, false, zero_value);

  test_saturating_div_overflow<I>(maxVal, I{1}, maxVal, false, zero_value);
  test_saturating_div_overflow<I>(maxVal, maxVal, I{1}, false, zero_value);

  // No saturation (no limit values)

  test_saturating_div_overflow<I>(I{27}, I{28}, I{0}, false, zero_value);
  test_saturating_div_overflow<I>(I{28}, I{27}, I{1}, false, zero_value);
  {
    constexpr I lesserVal = maxVal / I{2} + I{27};
    constexpr I biggerVal = maxVal / I{2} + I{28};
    test_saturating_div_overflow<I>(lesserVal, biggerVal, I{0}, false, zero_value);
    test_saturating_div_overflow<I>(biggerVal, lesserVal, I{1}, false, zero_value);
  }

  // Unsigned integer division never overflows
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

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
  static_assert(test(0));
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
  static_assert(test(0));

  return 0;
}
