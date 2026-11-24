//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
TEST_FUNC constexpr T invoke_bitmask(int start, int width)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(start);
    DoNotOptimize(width);
  }
  return cuda::bitmask<T>(start, width);
}

template <typename T>
TEST_FUNC constexpr bool test()
{
  using nl             = cuda::std::numeric_limits<T>;
  constexpr T all_ones = static_cast<T>(~T{0});

  assert(invoke_bitmask<T>(0, 0) == 0);
  assert(invoke_bitmask<T>(0, 1) == 1);
  assert(invoke_bitmask<T>(1, 0) == 0);
  assert(invoke_bitmask<T>(1, 1) == 0b10);
  assert(invoke_bitmask<T>(0, 2) == 0b11);
  assert(invoke_bitmask<T>(2, 2) == 0b1100);

  assert(invoke_bitmask<T>(0, 2) == 0b11);
  assert(invoke_bitmask<T>(3, 2) == 0b11000);
  assert(invoke_bitmask<T>(nl::digits - 1, 1) == (T{1} << (nl::digits - 1u)));
  assert(invoke_bitmask<T>(0, nl::digits) == all_ones);
  assert(invoke_bitmask<T>(nl::digits, 0) == 0);

  // Check every valid range, including empty masks at every start and ranges
  // starting, ending, or spanning across the 32-, 64-, and 96-bit boundaries.
  for (int start = 0; start <= nl::digits; ++start)
  {
    T expected = 0;
    assert(invoke_bitmask<T>(start, 0) == expected);
    for (int width = 1; width <= nl::digits - start; ++width)
    {
      // Build the reference one bit at a time without shifting by the type width.
      expected |= T{1} << (start + width - 1);
      assert(invoke_bitmask<T>(start, width) == expected);
    }
  }
  return true;
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
