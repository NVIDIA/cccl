//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// constexpr unspecified ignore;

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  { // Test that std::ignore provides constexpr converting assignment.
    auto& res = (cuda::std::ignore = 42);
    assert(&res == &cuda::std::ignore);
  }
  { // Test that cuda::std::ignore provides constexpr copy/move constructors
    auto copy  = cuda::std::ignore;
    auto moved = cuda::std::move(copy);
    unused(moved);
  }
  { // Test that cuda::std::ignore provides constexpr copy/move assignment
    auto copy  = cuda::std::ignore;
    copy       = cuda::std::ignore;
    auto moved = cuda::std::ignore;
    moved      = cuda::std::move(copy);
    unused(moved);
  }
  return true;
}
static_assert(cuda::std::is_trivial<decltype(cuda::std::ignore)>::value, "");

int main(int, char**)
{
  {
    constexpr auto& ignore_v = cuda::std::ignore;
    unused(ignore_v);
  }
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
