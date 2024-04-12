//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration_values::zero  // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "../../rep.h"
#include "test_macros.h"

int main(int, char**)
{
  assert(cuda::std::chrono::duration_values<int>::zero() == 0);
  assert(cuda::std::chrono::duration_values<Rep>::zero() == 0);
  static_assert(cuda::std::chrono::duration_values<int>::zero() == 0, "");
  static_assert(cuda::std::chrono::duration_values<Rep>::zero() == 0, "");

  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<int>::zero());
  LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<Rep>::zero());
#if TEST_STD_VER > 2017
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<int>::zero());
  ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<Rep>::zero());
#endif

  return 0;
}
