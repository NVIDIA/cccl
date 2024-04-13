//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

// UNSUPPORTED: c++98, c++03, c++11, c++14

// constexpr byte operator&(byte l, byte r) noexcept;

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b8{static_cast<cuda::std::byte>(8)};
  constexpr cuda::std::byte b9{static_cast<cuda::std::byte>(9)};

  static_assert(noexcept(b1 & b8), "");

  static_assert(cuda::std::to_integer<int>(b1 & b8) == 0, "");
  static_assert(cuda::std::to_integer<int>(b1 & b9) == 1, "");
  static_assert(cuda::std::to_integer<int>(b8 & b9) == 8, "");

  static_assert(cuda::std::to_integer<int>(b8 & b1) == 0, "");
  static_assert(cuda::std::to_integer<int>(b9 & b1) == 1, "");
  static_assert(cuda::std::to_integer<int>(b9 & b8) == 8, "");

  return 0;
}
