//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC constexpr void test_bit_trunc(T value, int width)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(value);
    DoNotOptimize(width);
  }

  T expected = 0;
  for (int bit = 0; bit < cuda::std::numeric_limits<T>::digits && bit < width; ++bit)
  {
    expected |= value & (T{1} << bit);
  }
  assert(cuda::bit_trunc(value, width) == expected);
}

template <class T>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::bit_trunc(T{0}, 0))>);
  static_assert(noexcept(cuda::bit_trunc(T{0}, 0)));

  constexpr int digits  = cuda::std::numeric_limits<T>::digits;
  constexpr T all_ones  = cuda::std::numeric_limits<T>::max();
  constexpr T high_bit  = static_cast<T>(T{1} << (digits - 1));
  constexpr T half_low  = static_cast<T>(all_ones >> (digits / 2));
  constexpr T half_high = static_cast<T>(all_ones << (digits / 2));
  constexpr T values[]  = {
    T{0},
    T{1},
    T{0b10},
    T{0b101},
    T{0b11001001},
    high_bit,
    half_low,
    half_high,
    all_ones,
    static_cast<T>(all_ones - T{1})};

  for (auto value : values)
  {
    for (int width = 0; width <= digits + 1; ++width)
    {
      test_bit_trunc(value, width);
    }
    // BZHI only reads the low eight bits of its width operand.
    test_bit_trunc(value, 255);
    test_bit_trunc(value, 256);
    test_bit_trunc(value, 257);
    test_bit_trunc(value, cuda::std::numeric_limits<int>::max());
  }

  for (int bit = 0; bit < digits; ++bit)
  {
    T value = static_cast<T>(T{1} << bit);
    test_bit_trunc(value, bit);
    test_bit_trunc(value, bit + 1);
  }
}

TEST_FUNC constexpr bool test()
{
  test<unsigned char>();
  test<unsigned short>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();
#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
