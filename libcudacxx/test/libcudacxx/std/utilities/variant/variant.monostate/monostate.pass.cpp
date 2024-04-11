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

// struct monostate {};

#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"

int main(int, char**)
{
  using M = cuda::std::monostate;
  static_assert(cuda::std::is_trivially_default_constructible<M>::value, "");
  static_assert(cuda::std::is_trivially_copy_constructible<M>::value, "");
  static_assert(cuda::std::is_trivially_copy_assignable<M>::value, "");
  static_assert(cuda::std::is_trivially_destructible<M>::value, "");
  constexpr M m{};
  ((void) m);

  return 0;
}
