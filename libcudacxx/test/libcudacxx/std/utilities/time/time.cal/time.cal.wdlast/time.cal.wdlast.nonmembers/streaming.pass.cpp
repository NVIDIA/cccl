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
// class weekday_last;

//   template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const weekday_last& wdl);
//
//   Returns: os << wdl.weekday() << "[last]".

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
  using weekday_last = cuda::std::chrono::weekday_last;
  using weekday      = cuda::std::chrono::weekday;

  std::cout << weekday_last{weekday{3}};

  return 0;
}
