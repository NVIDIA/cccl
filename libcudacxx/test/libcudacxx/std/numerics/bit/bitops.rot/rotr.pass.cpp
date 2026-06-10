//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class T>
//   constexpr int rotr(T x, unsigned int s) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//  T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
TEST_FUNC constexpr T expected_rotr(T value, int count)
{
  constexpr int digits = cuda::std::numeric_limits<T>::digits;
  int count_mod        = count % digits;
  if (count_mod < 0)
  {
    count_mod += digits;
  }
  return count_mod == 0 ? value : static_cast<T>((value >> count_mod) | (value << (digits - count_mod)));
}

template <typename T>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::rotr(T(0), 0))>);
  static_assert(noexcept(cuda::std::rotr(T(0), 0)));

  T values[] = {
    T(0),
    T(1),
    T(0xB3),
    cuda::std::numeric_limits<T>::max(),
    T{cuda::std::numeric_limits<T>::max() - 1},
  };
  for (const auto& value : values)
  {
    for (int count = -34; count <= 34; ++count)
    {
      assert(cuda::std::rotr(value, count) == expected_rotr(value, count));
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
  static_assert(test());
  assert(test());

  return 0;
}
