//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration_values::min  // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/limits>

#include "../../rep.h"
#include "test_macros.h"

int main(int, char**)
{
  assert(cuda::std::chrono::duration_values<int>::min() == cuda::std::numeric_limits<int>::lowest());
  assert(cuda::std::chrono::duration_values<double>::min() == cuda::std::numeric_limits<double>::lowest());
  assert(cuda::std::chrono::duration_values<Rep>::min() == cuda::std::numeric_limits<Rep>::lowest());
  static_assert(cuda::std::chrono::duration_values<int>::min() == cuda::std::numeric_limits<int>::lowest(), "");
  static_assert(cuda::std::chrono::duration_values<double>::min() == cuda::std::numeric_limits<double>::lowest(), "");
  static_assert(cuda::std::chrono::duration_values<Rep>::min() == cuda::std::numeric_limits<Rep>::lowest(), "");

  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<int>::min());
  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<double>::min());
  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<Rep>::min());
#if TEST_STD_VER > 2017
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<int>::min());
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<double>::min());
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<Rep>::min());
#endif

  return 0;
}
