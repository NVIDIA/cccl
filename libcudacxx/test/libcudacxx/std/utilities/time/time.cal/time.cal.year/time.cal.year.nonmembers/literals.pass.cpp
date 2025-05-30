//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr year operator""y(unsigned long long y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
#if _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()
  {
    using namespace cuda::std::chrono;
    static_assert(noexcept(4y));

    static_assert(2017y == year(2017));
    year y1 = 2018y;
    assert(y1 == year(2018));
  }

  {
    using namespace cuda::std::literals;
    static_assert(noexcept(4y));

    static_assert(2017y == cuda::std::chrono::year(2017));

    cuda::std::chrono::year y1 = 2020y;
    assert(y1 == cuda::std::chrono::year(2020));
  }
#endif // _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()

  return 0;
}
