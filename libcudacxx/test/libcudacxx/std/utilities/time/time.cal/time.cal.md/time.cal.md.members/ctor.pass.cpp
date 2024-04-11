//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class month_day;

//                     month_day() = default;
//  constexpr month_day(const chrono::month& m, const chrono::day& d) noexcept;
//
//  Effects:  Constructs an object of type month_day by initializing m_ with m, and d_ with d.
//
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using day       = cuda::std::chrono::day;
  using month     = cuda::std::chrono::month;
  using month_day = cuda::std::chrono::month_day;

  ASSERT_NOEXCEPT(month_day{});
  ASSERT_NOEXCEPT(month_day{month{1}, day{1}});

  constexpr month_day md0{};
  static_assert(md0.month() == month{}, "");
  static_assert(md0.day() == day{}, "");
  static_assert(!md0.ok(), "");

  constexpr month_day md1{cuda::std::chrono::January, day{4}};
  static_assert(md1.month() == cuda::std::chrono::January, "");
  static_assert(md1.day() == day{4}, "");
  static_assert(md1.ok(), "");

  return 0;
}
