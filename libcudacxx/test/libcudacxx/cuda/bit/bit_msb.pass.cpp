//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// bit_msb must only accept unsigned integer types, not bool, signed integers, floating point, or enums.
enum class Enum
{
  value
};

template <class T, class = void>
constexpr bool can_bit_msb = false;

template <class T>
constexpr bool can_bit_msb<T, cuda::std::void_t<decltype(cuda::bit_msb(cuda::std::declval<T>()))>> = true;

static_assert(can_bit_msb<unsigned char>);
static_assert(can_bit_msb<unsigned int>);
static_assert(can_bit_msb<unsigned long long>);
static_assert(!can_bit_msb<bool>);
static_assert(!can_bit_msb<int>);
static_assert(!can_bit_msb<float>);
static_assert(!can_bit_msb<Enum>);

template <typename T>
TEST_FUNC constexpr void test()
{
  using nl                              = cuda::std::numeric_limits<T>;
  [[maybe_unused]] constexpr T all_ones = static_cast<T>(~T{0});

  // a zero input has no set bit and returns -1
  assert(cuda::bit_msb(T{0}) == -1);
  // the most significant bit sits at index 0
  assert(cuda::bit_msb(T{1}) == 0);
  // a single bit set at position k returns k
  assert(cuda::bit_msb(static_cast<T>(T{1} << 3)) == 3);
  assert(cuda::bit_msb(static_cast<T>(T{1} << (nl::digits - 1))) == nl::digits - 1);
  // the highest set bit wins when several bits are set
  assert(cuda::bit_msb(static_cast<T>(0b10101000)) == 7);
  assert(cuda::bit_msb(all_ones) == nl::digits - 1);
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
  assert(test());
  static_assert(test());
  return 0;
}
