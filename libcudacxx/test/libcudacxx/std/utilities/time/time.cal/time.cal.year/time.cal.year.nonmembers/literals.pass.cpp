//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-5, clang-6, clang-7
// UNSUPPORTED: apple-clang-6, apple-clang-7, apple-clang-8, apple-clang-9
// UNSUPPORTED: apple-clang-10.0.0

// <chrono>
// class year;

// constexpr year operator""y(unsigned long long y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
#ifndef _LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS
  {
    using namespace cuda::std::chrono;
    ASSERT_NOEXCEPT(4y);

    static_assert(2017y == year(2017), "");
    year y1 = 2018y;
    assert(y1 == year(2018));
  }

  {
    using namespace cuda::std::literals;
    ASSERT_NOEXCEPT(4d);

    static_assert(2017y == cuda::std::chrono::year(2017), "");

    cuda::std::chrono::year y1 = 2020y;
    assert(y1 == cuda::std::chrono::year(2020));
  }
#endif // !_LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS

  return 0;
}
