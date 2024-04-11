//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// constexpr size_t variant_npos = -1;

#include <cuda/std/variant>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::variant_npos == static_cast<cuda::std::size_t>(-1), "");

  return 0;
}
