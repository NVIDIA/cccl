//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// constexpr common_type_t<duration> operator-() const;

#include <cuda/std/chrono>
#include <cuda/std/cassert>

#include "test_macros.h"

#pragma nv_diag_suppress set_but_not_used

int main(int, char**)
{
    {
    const cuda::std::chrono::minutes m(3);
    cuda::std::chrono::minutes m2 = -m;
    assert(m2.count() == -m.count());
    }
#if TEST_STD_VER >= 11
    {
    constexpr cuda::std::chrono::minutes m(3);
    constexpr cuda::std::chrono::minutes m2 = -m;
    static_assert(m2.count() == -m.count(), "");
    }
#endif

// P0548
    {
    typedef cuda::std::chrono::duration<int, cuda::std::ratio<10,10> > D10;
    typedef cuda::std::chrono::duration<int, cuda::std::ratio< 1, 1> > D1;
    D10 zero(0);
    D10 one(1);
    static_assert( (cuda::std::is_same< decltype(-one), decltype(zero-one) >::value), "");
    static_assert( (cuda::std::is_same< decltype(zero-one), D1>::value), "");
    static_assert( (cuda::std::is_same< decltype(-one),     D1>::value), "");
    static_assert( (cuda::std::is_same< decltype(+one),     D1>::value), "");
    }

  return 0;
}
