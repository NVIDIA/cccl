//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

// template <class IntegerType>
//    constexpr byte operator <<(byte b, IntegerType shift) noexcept;
// These functions shall not participate in overload resolution unless
//   is_integral_v<IntegerType> is true.

__host__ __device__ TEST_CONSTEXPR_CXX14 cuda::std::byte test(cuda::std::byte b)
{
  return b <<= 2;
}

int main(int, char**)
{
  constexpr cuda::std::byte b100{static_cast<cuda::std::byte>(100)};
  constexpr cuda::std::byte b115{static_cast<cuda::std::byte>(115)};

  static_assert(noexcept(b100 << 2), "");

  assert(cuda::std::to_integer<int>(b100 >> 1) == 50);
  assert(cuda::std::to_integer<int>(b100 >> 2) == 25);
  assert(cuda::std::to_integer<int>(b115 >> 3) == 14);
  assert(cuda::std::to_integer<int>(b115 >> 6) == 1);

#if TEST_STD_VER >= 2014
  static_assert(cuda::std::to_integer<int>(b100 >> 1) == 50, "");
  static_assert(cuda::std::to_integer<int>(b100 >> 2) == 25, "");
  static_assert(cuda::std::to_integer<int>(b115 >> 3) == 14, "");
  static_assert(cuda::std::to_integer<int>(b115 >> 6) == 1, "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
