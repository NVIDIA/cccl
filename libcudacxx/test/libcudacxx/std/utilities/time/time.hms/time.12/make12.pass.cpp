//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// constexpr hours make12(const hours& h) noexcept;
//   Returns: The 12-hour equivalent of h in the range [1h, 12h].
//     If h is not in the range [0h, 23h], the value returned is unspecified.

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

int main(int, char**)
{
  using hours = cuda::std::chrono::hours;
  static_assert(cuda::std::is_same_v<hours, decltype(cuda::std::chrono::make12(cuda::std::declval<hours>()))>);
  static_assert(noexcept(cuda::std::chrono::make12(cuda::std::declval<hours>())));

  static_assert(cuda::std::chrono::make12(hours(0)) == hours(12), "");
  static_assert(cuda::std::chrono::make12(hours(11)) == hours(11), "");
  static_assert(cuda::std::chrono::make12(hours(12)) == hours(12), "");
  static_assert(cuda::std::chrono::make12(hours(23)) == hours(11), "");

  assert(cuda::std::chrono::make12(hours(0)) == hours(12));
  for (int i = 1; i < 13; ++i)
  {
    assert(cuda::std::chrono::make12(hours(i)) == hours(i));
  }
  for (int i = 13; i < 24; ++i)
  {
    assert(cuda::std::chrono::make12(hours(i)) == hours(i - 12));
  }

  return 0;
}
