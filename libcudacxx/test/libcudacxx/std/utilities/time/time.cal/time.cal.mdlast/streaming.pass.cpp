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
// class month_day_last;
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day_last& mdl);
//
//     Returns: os << mdl.month() << "/last".

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using month_day_last = cuda::std::chrono::month_day_last;
  using month          = cuda::std::chrono::month;
  std::cout << month_day_last{month{1}};

  return 0;
}
