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

#include <cuda/std/cassert>
#include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_convertible.h"
#include "test_macros.h"
#include "variant_test_helpers.h"

int main(int, char**)
{
  // expected-error-re@variant:* 3 {{{{(static_assert|static assertion)}} failed}}
  cuda::std::variant<int, void> v; // expected-note {{requested here}}
  cuda::std::variant<int, const void> v2; // expected-note {{requested here}}
  cuda::std::variant<const volatile void, int> v3; // expected-note {{requested here}}

  return 0;
}
