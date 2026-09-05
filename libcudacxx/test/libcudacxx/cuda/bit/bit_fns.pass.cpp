//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// CONSTEXPR_STEPS: 15000000

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

// bit_fns must only accept unsigned integer types, not bool, signed integers, floating point, or enums.
enum class Enum
{
  value
};

template <class T>
_CCCL_CONCEPT can_bit_fns = _CCCL_REQUIRES_EXPR((T), T value)(cuda::bit_fns(value, 0));

static_assert(can_bit_fns<unsigned char>);
static_assert(can_bit_fns<unsigned int>);
static_assert(can_bit_fns<unsigned long long>);
static_assert(!can_bit_fns<bool>);
static_assert(!can_bit_fns<int>);
static_assert(!can_bit_fns<float>);
static_assert(!can_bit_fns<Enum>);

static_assert(cuda::std::is_same_v<decltype(cuda::bit_fns(1u, 0)), int>);
static_assert(noexcept(cuda::bit_fns(1u, 0)));

// Checks every rank in [0, num_bits) for one value. A linear scan from the least significant bit
// enumerates the set bits in rank order, so it yields the expected position for every rank without
// any search; every rank past the last set bit must return -1.
template <typename T>
TEST_FUNC constexpr bool check(T value)
{
  constexpr int digits = cuda::std::numeric_limits<T>::digits;
  int rank             = 0;
  for (int bit = 0; bit < digits; ++bit)
  {
    if (((value >> bit) & T{1}) != T{0})
    {
      if (cuda::bit_fns(value, rank++) != bit)
      {
        return false;
      }
    }
  }
  for (; rank < digits; ++rank)
  {
    if (cuda::bit_fns(value, rank) != -1)
    {
      return false;
    }
  }
  return true;
}

template <typename T>
TEST_FUNC constexpr void test()
{
  constexpr int digits = cuda::std::numeric_limits<T>::digits;
  constexpr int half   = digits / 2;
  constexpr T all_ones = static_cast<T>(~T{0});

  // a single set bit at any position has rank 0 and reports that position
  for (int bit = 0; bit < digits; ++bit)
  {
    assert(cuda::bit_fns(static_cast<T>(T{1} << bit), 0) == bit);
  }
  // with every bit set, rank k is at position k
  for (int rank = 0; rank < digits; ++rank)
  {
    assert(cuda::bit_fns(all_ones, rank) == rank);
  }
  // 0b10110100 has set bits at positions 2, 4, 5, and 7; there is no set bit of rank 4
  assert(cuda::bit_fns(static_cast<T>(0b10110100), 0) == 2);
  assert(cuda::bit_fns(static_cast<T>(0b10110100), 1) == 4);
  assert(cuda::bit_fns(static_cast<T>(0b10110100), 2) == 5);
  assert(cuda::bit_fns(static_cast<T>(0b10110100), 3) == 7);
  assert(cuda::bit_fns(static_cast<T>(0b10110100), 4) == -1);

  // every rank of: zero and one (all ranks not found, or all but the first), alternating bits,
  // each half of the value, the two extremes, and a pair straddling the first split
  assert(check(T{0}));
  assert(check(T{1}));
  assert(check(all_ones));
  assert(check(static_cast<T>(all_ones / 3))); // 0b0101...
  assert(check(static_cast<T>(all_ones - all_ones / 3))); // 0b1010...
  assert(check(static_cast<T>(all_ones >> half)));
  assert(check(static_cast<T>(all_ones << half)));
  assert(check(static_cast<T>(T{1} | static_cast<T>(T{1} << (digits - 1)))));
  assert(check(static_cast<T>(static_cast<T>(T{1} << (half - 1)) | static_cast<T>(T{1} << half))));
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
