//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class T>
//   constexpr int rotl(T x, unsigned int s) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//  T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr T expected_rotl(T value, int count)
{
  constexpr int digits = cuda::std::numeric_limits<T>::digits;
  int count_mod        = count % digits;
  if (count_mod < 0)
  {
    count_mod += digits;
  }

  T result = 0;
  for (int dst = 0; dst < digits; ++dst)
  {
    int src = (dst - count_mod + digits) % digits;
    if ((value & (T{1} << src)) != T{0})
    {
      result = result | (T{1} << dst);
    }
  }
  return result;
}

template <class T>
TEST_FUNC constexpr T invoke_rotl(T value, int count)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    DoNotOptimize(value);
    DoNotOptimize(count);
  }
  return cuda::std::rotl(value, count);
}

template <typename T>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::rotl(T(0), 0))>);
  static_assert(noexcept(cuda::std::rotl(T(0), 0)));

  T values[] = {
    T{0},
    T{1},
    T{0xB3},
    cuda::std::numeric_limits<T>::max(),
    T{cuda::std::numeric_limits<T>::max() - 1},
  };
  for (const auto& value : values)
  {
    for (int count = -34; count <= 34; ++count)
    {
      auto rotated = invoke_rotl(value, count);
      assert(rotated == expected_rotl(value, count));
      assert(rotated == cuda::std::rotr(value, -count));
      assert(cuda::std::rotr(rotated, count) == value);
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
  static_assert(test());
  assert(test());

  return 0;
}
