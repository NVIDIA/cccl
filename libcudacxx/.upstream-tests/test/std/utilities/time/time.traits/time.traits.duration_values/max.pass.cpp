//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration_values::max  // noexcept after C++17

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "../../rep.h"

#include <cuda/std/cstdint>
#ifndef __device__
#error whomp whomp
#endif

int main(int, char**)
{
    assert(cuda::std::chrono::duration_values<int>::max() ==
           cuda::std::numeric_limits<int>::max());
    assert(cuda::std::chrono::duration_values<double>::max() ==
           cuda::std::numeric_limits<double>::max());
    assert(cuda::std::chrono::duration_values<Rep>::max() ==
           cuda::std::numeric_limits<Rep>::max());
#if TEST_STD_VER >= 11
    static_assert(cuda::std::chrono::duration_values<int>::max() ==
           cuda::std::numeric_limits<int>::max(), "");
    static_assert(cuda::std::chrono::duration_values<double>::max() ==
           cuda::std::numeric_limits<double>::max(), "");
    static_assert(cuda::std::chrono::duration_values<Rep>::max() ==
           cuda::std::numeric_limits<Rep>::max(), "");
#endif

    LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<int>::max());
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<double>::max());
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<Rep>::max());
#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<int>::max());
    ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<double>::max());
    ASSERT_NOEXCEPT(cuda::std::chrono::duration_values<Rep>::max());
#endif

  return 0;
}
