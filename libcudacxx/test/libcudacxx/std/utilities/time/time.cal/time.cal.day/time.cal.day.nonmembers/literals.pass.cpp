//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class day;

// constexpr day operator""d(unsigned long long d) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
#if _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()
  {
    using namespace cuda::std::chrono;
    static_assert(noexcept(4d));
    static_assert(cuda::std::is_same_v<day, decltype(4d)>);

    static_assert(7d == day(7));
    day d1 = 4d;
    assert(d1 == day(4));
  }

  {
    using namespace cuda::std::literals;
    static_assert(noexcept(4d));
    static_assert(cuda::std::is_same_v<cuda::std::chrono::day, decltype(4d)>);

    static_assert(7d == cuda::std::chrono::day(7));

    cuda::std::chrono::day d1 = 4d;
    assert(d1 == cuda::std::chrono::day(4));
  }
#endif // _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS()

  return 0;
}
