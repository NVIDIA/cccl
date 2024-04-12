//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: *

// <chrono>
// class month_weekday;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_weekday& mwd);
//
//     Returns: os << mwd.month() << '/' << mwd.weekday_indexed().

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday   = cuda::std::chrono::month_weekday;
  using month           = cuda::std::chrono::month;
  using weekday_indexed = cuda::std::chrono::weekday_indexed;
  using weekday         = cuda::std::chrono::weekday;

  std::cout << month_weekday{month{1}, weekday_indexed{weekday{3}, 3}};

  return 0;
}
