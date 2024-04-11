//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_day_last;

// constexpr operator sys_days() const noexcept;
//  Returns: sys_days{year()/month()/day()}.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                = cuda::std::chrono::year;
  using month_day_last      = cuda::std::chrono::month_day_last;
  using year_month_day_last = cuda::std::chrono::year_month_day_last;
  using sys_days            = cuda::std::chrono::sys_days;
  using days                = cuda::std::chrono::days;

  ASSERT_NOEXCEPT(static_cast<sys_days>(cuda::std::declval<const year_month_day_last>()));
  ASSERT_SAME_TYPE(sys_days, decltype(static_cast<sys_days>(cuda::std::declval<const year_month_day_last>())));

  auto constexpr January  = cuda::std::chrono::January;
  auto constexpr November = cuda::std::chrono::November;

  { // Last day in Jan 1970 was the 31st
    constexpr year_month_day_last ymdl{year{1970}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{30}, "");
  }

  {
    constexpr year_month_day_last ymdl{year{2000}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{10957 + 30}, "");
  }

  {
    constexpr year_month_day_last ymdl{year{1940}, month_day_last{January}};
    constexpr sys_days sd{ymdl};

    static_assert(sd.time_since_epoch() == days{-10957 + 29}, "");
  }

  {
    year_month_day_last ymdl{year{1939}, month_day_last{November}};
    sys_days sd{ymdl};

    assert(sd.time_since_epoch() == days{-(10957 + 33)});
  }

  return 0;
}
