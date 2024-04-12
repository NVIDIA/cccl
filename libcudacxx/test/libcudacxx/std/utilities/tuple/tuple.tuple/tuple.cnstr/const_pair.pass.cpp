//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class U1, class U2> tuple(const pair<U1, U2>& u);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<long, char> T0;
    typedef cuda::std::tuple<long long, short> T1;
    T0 t0(2, 'a');
    T1 t1 = t0;
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == short('a'));
  }
#if TEST_STD_VER > 2011
  {
    typedef cuda::std::pair<long, char> P0;
    typedef cuda::std::tuple<long long, short> T1;
    constexpr P0 p0(2, 'a');
    constexpr T1 t1 = p0;
    static_assert(cuda::std::get<0>(t1) == cuda::std::get<0>(p0), "");
    static_assert(cuda::std::get<1>(t1) == cuda::std::get<1>(p0), "");
    static_assert(cuda::std::get<0>(t1) == 2, "");
    static_assert(cuda::std::get<1>(t1) == short('a'), "");
  }
#endif

  return 0;
}
