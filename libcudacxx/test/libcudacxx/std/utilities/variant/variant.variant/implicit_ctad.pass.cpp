//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// Make sure that the implicitly-generated CTAD works.

// We make sure that it is not ill-formed, however we still produce a warning for
// this one because explicit construction from a variant using CTAD is ambiguous
// (in the sense that the programer intent is not clear).
// ADDITIONAL_COMPILE_OPTIONS_HOST: -Wno-ctad-maybe-unsupported

#include <cuda/std/variant>

#include "test_macros.h"

int main(int, char**)
{
  // This is the motivating example from P0739R0
  {
    cuda::std::variant<int, double> v1(3);
    cuda::std::variant v2 = v1;
    static_assert(cuda::std::is_same_v<decltype(v2), cuda::std::variant<int, double>>);
    unused(v2);
  }

  {
    cuda::std::variant<int, double> v1(3);
    cuda::std::variant v2 = cuda::std::variant(v1); // Technically valid, but intent is ambiguous!
    static_assert(cuda::std::is_same_v<decltype(v2), cuda::std::variant<int, double>>);
    unused(v2);
  }

  return 0;
}
