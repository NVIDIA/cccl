//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class weekday_indexed;

// constexpr bool operator==(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: x.weekday() == y.weekday() && x.index() == y.index().
// constexpr bool operator!=(const weekday_indexed& x, const weekday_indexed& y) noexcept;
//   Returns: !(x == y)

#include <cuda/std/chrono>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using weekday         = cuda::std::chrono::weekday;
    using weekday_indexed = cuda::std::chrono::weekday_indexed;

    AssertComparisons2AreNoexcept<weekday_indexed>();
    AssertComparisons2ReturnBool<weekday_indexed>();

    static_assert( (weekday_indexed{} == weekday_indexed{}), "");
    static_assert(!(weekday_indexed{} != weekday_indexed{}), "");

    static_assert(!(weekday_indexed{} == weekday_indexed{cuda::std::chrono::Tuesday, 1}), "");
    static_assert( (weekday_indexed{} != weekday_indexed{cuda::std::chrono::Tuesday, 1}), "");

//  Some 'ok' values as well
    static_assert( (weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 2}), "");
    static_assert(!(weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 2}), "");

    static_assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{1}, 1}), "");
    static_assert( (weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{1}, 1}), "");
    static_assert(!(weekday_indexed{weekday{1}, 2} == weekday_indexed{weekday{2}, 2}),  "");
    static_assert( (weekday_indexed{weekday{1}, 2} != weekday_indexed{weekday{2}, 2}),  "");

  return 0;
}
