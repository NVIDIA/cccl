//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Regression test: the base 10 width of exact powers of ten used to be
// undercounted by one, writing one byte before the output buffer while
// reporting success

#include <cuda/std/charconv>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr int digit_count(T value)
{
  // counts the characters to_chars writes: one more for the sign when negative.
  // dividing the magnitude by ten before negating keeps the minimum representable,
  // and that division drops exactly one digit, which the second increment accounts for
  int count             = 1;
  T remaining           = value;
  constexpr bool is_neg = cuda::std::is_signed_v<T>;
  if constexpr (is_neg)
  {
    if (remaining < 0)
    {
      remaining = static_cast<T>(-(remaining / T{10}));
      ++count;
    }
  }
  while (remaining >= T{10})
  {
    remaining /= T{10};
    ++count;
  }
  return count;
}

template <typename T>
TEST_FUNC constexpr bool test_value(T value)
{
  const int expected_digits = digit_count(value);
  const int expected_len    = value < 0 ? expected_digits + 1 : expected_digits;

  char raw[256] = {};
  char* buf     = raw + 1;
  raw[0]        = '#';

  auto res = cuda::std::to_chars(buf, buf + 254, value);
  if (res.ec != cuda::std::errc{})
  {
    return false;
  }
  if (res.ptr - buf != expected_len)
  {
    return false;
  }
  if (raw[0] != '#')
  {
    return false;
  }

  // round trip
  T parsed{};
  auto pr = cuda::std::from_chars(buf, res.ptr, parsed);
  if (pr.ec != cuda::std::errc{} || parsed != value)
  {
    return false;
  }
  return true;
}

template <typename T>
TEST_FUNC constexpr bool test_powers_of_ten()
{
  T power = T{10};
  while (power <= cuda::std::numeric_limits<T>::max() / T{10})
  {
    if (!test_value(power))
    {
      return false;
    }
    if (!test_value(static_cast<T>(-power)))
    {
      return false;
    }
    power *= T{10};
  }
  return test_value(power);
}

int main(int, char**)
{
  assert(test_powers_of_ten<int32_t>());
  static_assert(test_powers_of_ten<uint32_t>());

  assert(test_powers_of_ten<uint32_t>());
  assert(test_powers_of_ten<int64_t>());
  assert(test_powers_of_ten<uint64_t>());
#if _CCCL_HAS_INT128()
  assert(test_powers_of_ten<__int128_t>());
  assert(test_powers_of_ten<__uint128_t>());
#endif // _CCCL_HAS_INT128()

  return 0;
}
