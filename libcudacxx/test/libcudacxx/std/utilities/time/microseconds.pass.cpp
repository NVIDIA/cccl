//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// using microseconds = duration<signed integral type of at least 55 bits, micro>;

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

int main(int, char**)
{
  using D      = cuda::std::chrono::microseconds;
  using Rep    = D::rep;
  using Period = D::period;
  static_assert(cuda::std::is_signed<Rep>::value, "");
  static_assert(cuda::std::is_integral<Rep>::value, "");
  static_assert(cuda::std::numeric_limits<Rep>::digits >= 54, "");
  static_assert((cuda::std::is_same<Period, cuda::std::micro>::value), "");

  return 0;
}
