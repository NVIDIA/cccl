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

// template <class ...Types> class variant;

#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

int main(int, char**)
{
  // expected-error-re@variant:* 1 {{{{(static_assert|static assertion)}} failed}}
  cuda::std::variant<> v; // expected-note {{requested here}}

  return 0;
}
