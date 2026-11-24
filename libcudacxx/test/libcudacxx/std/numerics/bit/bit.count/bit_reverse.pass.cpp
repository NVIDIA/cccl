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
//   constexpr T bit_reverse(T value) noexcept;

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr T expected_bit_reverse(T value)
{
  T result = 0;
  for (int i = 0; i < cuda::std::numeric_limits<T>::digits; ++i)
  {
    result <<= 1;
    result |= value & 1;
    value >>= 1;
  }
  return result;
}

template <typename T>
TEST_FUNC constexpr void test_bit_reverse(T value)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(value);
  }
  assert(cuda::std::bit_reverse(value) == expected_bit_reverse(value));
  assert(cuda::std::bit_reverse(cuda::std::bit_reverse(value)) == value);
}

template <typename T>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::bit_reverse(T{0}))>);
  static_assert(noexcept(cuda::std::bit_reverse(T{0})));

  constexpr int digits  = cuda::std::numeric_limits<T>::digits;
  constexpr T all_ones  = cuda::std::numeric_limits<T>::max();
  constexpr T low_half  = static_cast<T>(all_ones >> (digits / 2));
  constexpr T high_half = static_cast<T>(all_ones ^ low_half);
  constexpr T high_bit  = static_cast<T>(T{1} << (digits - 1));

  constexpr T values[] = {
    T{0},
    T{1},
    T{2},
    T{0b11001001},
    high_bit,
    low_half,
    high_half,
    all_ones,
    static_cast<T>(all_ones - T{1}),
  };

  for (auto value : values)
  {
    test_bit_reverse(value);
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
