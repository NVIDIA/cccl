//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include <test_macros.h>

int main(int, char**)
{
  constexpr cuda::std::byte b{42};
  static_assert(cuda::std::to_integer<int>(b) == 42, "");

  return 0;
}
