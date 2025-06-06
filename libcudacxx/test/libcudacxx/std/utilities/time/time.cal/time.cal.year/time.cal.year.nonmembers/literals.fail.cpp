//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++17

// <chrono>
// class year;

// constexpr year operator""y(unsigned long long y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using cuda::std::chrono::year;
  year d1 = 1234y; // expected-error-re {{no matching literal operator for call to 'operator""y' {{.*}}}}

  return 0;
}
