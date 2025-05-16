//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++17

// <chrono>
// class day;

// constexpr day operator""d(unsigned long long d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using day = cuda::std::chrono::day;
  day d1    = 4d; // expected-error-re {{no matching literal operator for call to 'operator""d' {{.*}}}}

  return 0;
}
