//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <cuda/std/chrono>

// using months = duration<signed integer type of at least 20 bits, ratio_divide<years::period, ratio<12>>>;

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

int main(int, char**)
{
  typedef cuda::std::chrono::months D;
  typedef D::rep Rep;
  typedef D::period Period;
  static_assert(cuda::std::is_signed<Rep>::value, "");
  static_assert(cuda::std::is_integral<Rep>::value, "");
  static_assert(cuda::std::numeric_limits<Rep>::digits >= 20, "");
  static_assert(
    cuda::std::is_same_v<Period, cuda::std::ratio_divide<cuda::std::chrono::years::period, cuda::std::ratio<12>>>, "");

  return 0;
}
