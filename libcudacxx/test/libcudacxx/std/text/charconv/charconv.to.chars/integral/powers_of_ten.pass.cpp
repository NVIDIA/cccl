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

#include <cuda/std/cassert>
#include <cuda/std/charconv>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr void test_value(T value, int expected_digits)
{
  int expected_len = expected_digits;
  if constexpr (cuda::std::is_signed_v<T>)
  {
    if (value < 0)
    {
      ++expected_len;
    }
  }

  char raw[256] = {};
  char* buf     = raw + 1;
  raw[0]        = '#';

  auto res = cuda::std::to_chars(buf, buf + 254, value);
  assert(res.ec == cuda::std::errc{});
  assert(res.ptr - buf == expected_len);
  assert(raw[0] == '#');

  // a buffer one character short of the exact width must fail without writing
  auto narrow = cuda::std::to_chars(buf, buf + expected_len - 1, value);
  assert(narrow.ec == cuda::std::errc::value_too_large);
  assert(raw[0] == '#');

  T parsed{};
  auto pr = cuda::std::from_chars(buf, res.ptr, parsed);
  assert(pr.ec == cuda::std::errc{});
  assert(parsed == value);
}

template <typename T>
TEST_FUNC constexpr void test_powers_of_ten()
{
  T power    = T{10};
  int digits = 2;
  while (power <= cuda::std::numeric_limits<T>::max() / T{10})
  {
    test_value(power, digits);
    if constexpr (cuda::std::is_signed_v<T>)
    {
      test_value(-power, digits);
    }
    power *= T{10};
    ++digits;
  }
  test_value(power, digits);
  if constexpr (cuda::std::is_signed_v<T>)
  {
    test_value(-power, digits);
  }
}

TEST_FUNC constexpr bool test()
{
  test_powers_of_ten<cuda::std::int8_t>();
  test_powers_of_ten<cuda::std::int16_t>();
  test_powers_of_ten<cuda::std::int32_t>();
  test_powers_of_ten<cuda::std::int64_t>();
#if _CCCL_HAS_INT128()
  test_powers_of_ten<__int128_t>();
#endif // _CCCL_HAS_INT128()

  test_powers_of_ten<cuda::std::uint8_t>();
  test_powers_of_ten<cuda::std::uint16_t>();
  test_powers_of_ten<cuda::std::uint32_t>();
  test_powers_of_ten<cuda::std::uint64_t>();
#if _CCCL_HAS_INT128()
  test_powers_of_ten<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
