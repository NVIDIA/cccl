//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// typedef duration<signed integral type of at least 45 bits, milli> milliseconds;

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

int main(int, char**)
{
  typedef cuda::std::chrono::milliseconds D;
  typedef D::rep Rep;
  typedef D::period Period;
  static_assert(cuda::std::is_signed<Rep>::value, "");
  static_assert(cuda::std::is_integral<Rep>::value, "");
  static_assert(cuda::std::numeric_limits<Rep>::digits >= 44, "");
  static_assert((cuda::std::is_same<Period, cuda::std::milli>::value), "");

  return 0;
}
