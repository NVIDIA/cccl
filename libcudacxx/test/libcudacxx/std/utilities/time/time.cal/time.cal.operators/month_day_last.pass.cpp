//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_day_last;

// constexpr month_day_last
//   operator/(const month& m, last_spec) noexcept;
// Returns: month_day_last{m}.
//
// constexpr month_day_last
//   operator/(int m, last_spec) noexcept;
// Returns: month(m) / last.
//
// constexpr month_day_last
//   operator/(last_spec, const month& m) noexcept;
// Returns: m / last.
//
// constexpr month_day_last
//   operator/(last_spec, int m) noexcept;
// Returns: month(m) / last.
//
//
// [Note: A month_day_last object can be constructed using the expression m/last or last/m,
//     where m is an expression of type month. — end note]
// [Example:
//     constexpr auto mdl = February/last; // mdl is the last day of February of an as yet unspecified year
//     static_assert(mdl.month() == February);
// --end example]

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month          = cuda::std::chrono::month;
  using month_day_last = cuda::std::chrono::month_day_last;

  constexpr month February                    = cuda::std::chrono::February;
  constexpr cuda::std::chrono::last_spec last = cuda::std::chrono::last;

  static_assert(cuda::std::is_same_v<month_day_last, decltype(last / February)>);
  static_assert(cuda::std::is_same_v<month_day_last, decltype(February / last)>);

  //  Run the example
  {
    constexpr auto mdl = February / cuda::std::chrono::last;
    static_assert(mdl.month() == February, "");
  }

  { // operator/(const month& m, last_spec) and switched
    static_assert(noexcept(last / February));
    static_assert(cuda::std::is_same_v<month_day_last, decltype(last / February)>);
    static_assert(noexcept(February / last));
    static_assert(cuda::std::is_same_v<month_day_last, decltype(February / last)>);

    static_assert((last / February).month() == February, "");
    static_assert((February / last).month() == February, "");

    for (unsigned i = 1; i < 12; ++i)
    {
      month m{i};
      month_day_last mdl1 = last / m;
      month_day_last mdl2 = m / last;
      assert(mdl1.month() == m);
      assert(mdl2.month() == m);
      assert(mdl1 == mdl2);
    }
  }

  { // operator/(int, last_spec) and switched
    static_assert(noexcept(last / 2));
    static_assert(cuda::std::is_same_v<month_day_last, decltype(last / 2)>);
    static_assert(noexcept(2 / last));
    static_assert(cuda::std::is_same_v<month_day_last, decltype(2 / last)>);

    static_assert((last / 2).month() == February, "");
    static_assert((2 / last).month() == February, "");

    for (unsigned i = 1; i < 12; ++i)
    {
      month m{i};
      month_day_last mdl1 = last / i;
      month_day_last mdl2 = i / last;
      assert(mdl1.month() == m);
      assert(mdl2.month() == m);
      assert(mdl1 == mdl2);
    }
  }

  return 0;
}
