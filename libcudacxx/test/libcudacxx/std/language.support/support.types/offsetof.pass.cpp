//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>

#include "test_macros.h"

#ifndef offsetof
#  error offsetof not defined
#endif

struct WithAddressOperator
{
  TEST_FUNC int operator&();
};

struct A
{
  int x, y;
  char z;
  WithAddressOperator with_address_op;
};

int main(int, char**)
{
  static_assert(noexcept(offsetof(A, x)));
  static_assert(offsetof(A, x) == 0);
  static_assert(offsetof(A, y) == sizeof(A::x));
  static_assert(offsetof(A, z) == sizeof(A::x) + sizeof(A::y));

  // Querying offsetof of members that implement operator& requires __builtin_offsetof or __builtin_addressof for nvrtc
  // implementation (available since 13.3).
#if !TEST_COMPILER(NVRTC, <, 12, 3)
  static_assert(offsetof(A, with_address_op) != 0);
#endif // !TEST_COMPILER(NVRTC, <, 12, 3)

  return 0;
}
