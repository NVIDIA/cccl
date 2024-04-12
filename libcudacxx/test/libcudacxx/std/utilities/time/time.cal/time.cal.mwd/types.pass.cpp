//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_weekday;

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using month_weekday = cuda::std::chrono::month_weekday;

  static_assert(cuda::std::is_trivially_copyable_v<month_weekday>, "");
  static_assert(cuda::std::is_standard_layout_v<month_weekday>, "");

  return 0;
}
