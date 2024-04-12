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

// constexpr byte operator~(byte b) noexcept;

int main(int, char**)
{
  constexpr cuda::std::byte b1{static_cast<cuda::std::byte>(1)};
  constexpr cuda::std::byte b2{static_cast<cuda::std::byte>(2)};
  constexpr cuda::std::byte b8{static_cast<cuda::std::byte>(8)};

  static_assert(noexcept(~b1), "");

  static_assert(cuda::std::to_integer<int>(~b1) == 254, "");
  static_assert(cuda::std::to_integer<int>(~b2) == 253, "");
  static_assert(cuda::std::to_integer<int>(~b8) == 247, "");

  return 0;
}
