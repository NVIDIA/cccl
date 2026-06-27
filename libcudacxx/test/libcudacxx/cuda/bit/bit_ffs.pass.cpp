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
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

// bit_ffs must only accept unsigned integer types, not bool, signed integers, floating point, or enums.
enum class Enum
{
  value
};

template <class T>
_CCCL_CONCEPT can_bit_ffs = _CCCL_REQUIRES_EXPR((T), T value)(cuda::bit_ffs(value));

static_assert(can_bit_ffs<unsigned char>);
static_assert(can_bit_ffs<unsigned int>);
static_assert(can_bit_ffs<unsigned long long>);
static_assert(!can_bit_ffs<bool>);
static_assert(!can_bit_ffs<int>);
static_assert(!can_bit_ffs<float>);
static_assert(!can_bit_ffs<Enum>);

template <typename T>
TEST_FUNC constexpr void test()
{
  using nl                              = cuda::std::numeric_limits<T>;
  [[maybe_unused]] constexpr T all_ones = static_cast<T>(~T{0});

  // a zero input is well defined and returns 0
  assert(cuda::bit_ffs(T{0}) == 0);
  // the least significant bit set maps to position 1
  assert(cuda::bit_ffs(T{1}) == 1);
  assert(cuda::bit_ffs(all_ones) == 1);
  // a single bit set at position k maps to result k + 1
  assert(cuda::bit_ffs(static_cast<T>(T{1} << 1)) == 2);
  assert(cuda::bit_ffs(static_cast<T>(T{1} << 3)) == 4);
  assert(cuda::bit_ffs(static_cast<T>(T{1} << (nl::digits - 1))) == nl::digits);
  // the lowest set bit wins when several bits are set
  assert(cuda::bit_ffs(static_cast<T>(0b10101000)) == 4);
}

TEST_FUNC constexpr bool test()
{
  test<unsigned char>();
  test<unsigned short>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();

  test<uint8_t>();
  test<uint16_t>();
  test<uint32_t>();
  test<uint64_t>();
  test<size_t>();
  test<uintmax_t>();
  test<uintptr_t>();

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
