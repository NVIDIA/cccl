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
//   constexpr T bit_compress(T value, T mask) noexcept;

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_macros.h"

// nvcc complains about the u128 literal constants being too large
_CCCL_DIAG_SUPPRESS_NVCC(23)

template <class T>
TEST_FUNC constexpr T expected_bit_compress(T value, T mask)
{
  T result = 0;
  for (int i = 0, j = 0; i < cuda::std::numeric_limits<T>::digits; ++i)
  {
    bool mask_bit = (mask >> i) & 1;
    result |= (static_cast<T>(mask_bit) & (value >> i)) << j;
    j += mask_bit;
  }
  return result;
}

template <class T>
TEST_FUNC constexpr void test_bit_compress(T value, T mask)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(value);
    DoNotOptimize(mask);
  }
  assert(cuda::std::bit_compress(value, mask) == expected_bit_compress(value, mask));
}

template <class T>
TEST_FUNC constexpr void test()
{
  using namespace test_integer_literals;

  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::bit_compress(T{0}, T{0}))>);
  static_assert(noexcept(cuda::std::bit_compress(T{0}, T{0})));

  constexpr int digits  = cuda::std::numeric_limits<T>::digits;
  constexpr T all_ones  = cuda::std::numeric_limits<T>::max();
  constexpr T low_half  = static_cast<T>(all_ones >> (digits / 2));
  constexpr T high_half = static_cast<T>(all_ones ^ low_half);
  constexpr T high_bit  = static_cast<T>(T{1} << (digits - 1));

  constexpr T values[] = {
    T{0},
    T{1},
#if _CCCL_HAS_INT128()
    static_cast<T>(0xb3b3'b3b3'b3b3'b3b3'b3b3'b3b3'b3b3'b3b3_u128),
    static_cast<T>(0x7b1a'7b1a'7b1a'7b1a'7b1a'7b1a'7b1a'7b1a_u128),
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    static_cast<T>(0xb3b3'b3b3'b3b3'b3b3),
    static_cast<T>(0x7b1a'7b1a'7b1a'7b1a),
#endif // ^^^ !_CCCL_HAS_INT128() ^^^
    high_bit,
    low_half,
    high_half,
    all_ones,
    static_cast<T>(all_ones - T{1}),
  };
  constexpr T masks[] = {
    T{0},
    T{1},
#if _CCCL_HAS_INT128()
    static_cast<T>(0x5555'5555'5555'5555'5555'5555'5555'5555_u128),
    static_cast<T>(0xaaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa'aaaa_u128),
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
    static_cast<T>(0x5555'5555'5555'5555),
    static_cast<T>(0xaaaa'aaaa'aaaa'aaaa),
#endif // ^^^ !_CCCL_HAS_INT128() ^^^
    high_bit,
    low_half,
    high_half,
    all_ones,
  };

  for (auto value : values)
  {
    for (auto mask : masks)
    {
      test_bit_compress(value, mask);
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
