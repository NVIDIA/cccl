//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class T>
//   constexpr T bit_repeat(T value, int width) noexcept;

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC constexpr T expected_bit_repeat(T value, int width)
{
  T result = 0;
  for (int i = 0; i < cuda::std::numeric_limits<T>::digits; ++i)
  {
    result |= ((value >> (i % width)) & 1) << i;
  }
  return result;
}

template <class T>
TEST_FUNC constexpr void test_bit_repeat(T value, int width)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(value);
    DoNotOptimize(width);
  }
  assert(cuda::std::bit_repeat(value, width) == expected_bit_repeat(value, width));
}

template <class T>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::bit_repeat(T{0}, 1))>);

  // nvcc 12.0 doesn't evaluate noexcept correctly.
#if !TEST_CUDA_COMPILER(NVCC, ==, 12, 0)
  static_assert(!noexcept(cuda::std::bit_repeat(T{0}, 1)));
#endif // !TEST_CUDA_COMPILER(NVCC, ==, 12, 0)

  constexpr int digits = cuda::std::numeric_limits<T>::digits;
  constexpr T all_ones = cuda::std::numeric_limits<T>::max();
  constexpr T high_bit = static_cast<T>(T{1} << (digits - 1));
  constexpr T values[] = {
    T{0},
    T{1},
    T{0b10},
    T{0b101},
    T{0b11001001},
    high_bit,
    all_ones,
    static_cast<T>(all_ones - T{1}),
  };

  for (auto value : values)
  {
    for (int width = 1; width <= digits + 1; ++width)
    {
      test_bit_repeat(value, width);
    }
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
