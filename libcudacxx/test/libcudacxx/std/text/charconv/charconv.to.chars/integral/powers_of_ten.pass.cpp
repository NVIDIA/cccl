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
TEST_FUNC constexpr bool test_value(T value, int expected_digits)
{
  const int expected_len = value < 0 ? expected_digits + 1 : expected_digits;

  char raw[256] = {};
  char* buf     = raw + 1;
  raw[0]        = '#';

  auto res = cuda::std::to_chars(buf, buf + 254, value);
  if (res.ec != cuda::std::errc{} || res.ptr - buf != expected_len || raw[0] != '#')
  {
    return false;
  }

  // a buffer one character short of the exact width must fail without writing
  auto narrow = cuda::std::to_chars(buf, buf + expected_len - 1, value);
  if (narrow.ec != cuda::std::errc::value_too_large)
  {
    return false;
  }

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
  T power    = T{10};
  int digits = 2;
  while (power <= cuda::std::numeric_limits<T>::max() / T{10})
  {
    if (!test_value(power, digits))
    {
      return false;
    }
    if constexpr (cuda::std::is_signed_v<T>)
    {
      if (!test_value(T{-1} * power, digits + 1))
      {
        return false;
      }
    }
    power *= T{10};
    ++digits;
  }
  if (!test_value(power, digits))
  {
    return false;
  }
  if constexpr (cuda::std::is_signed_v<T>)
  {
    return test_value(T{-1} * power, digits + 1);
  }
  return true;
}

TEST_FUNC constexpr bool test()
{
  bool ok = true;
  ok = ok && test_powers_of_ten<cuda::std::int8_t>();
  ok = ok && test_powers_of_ten<cuda::std::int16_t>();
  ok = ok && test_powers_of_ten<cuda::std::int32_t>();
  ok = ok && test_powers_of_ten<cuda::std::int64_t>();
#if _CCCL_HAS_INT128()
  ok = ok && test_powers_of_ten<__int128_t>();
#endif // _CCCL_HAS_INT128()

  ok = ok && test_powers_of_ten<cuda::std::uint8_t>();
  ok = ok && test_powers_of_ten<cuda::std::uint16_t>();
  ok = ok && test_powers_of_ten<cuda::std::uint32_t>();
  ok = ok && test_powers_of_ten<cuda::std::uint64_t>();
#if _CCCL_HAS_INT128()
  ok = ok && test_powers_of_ten<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return ok;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
