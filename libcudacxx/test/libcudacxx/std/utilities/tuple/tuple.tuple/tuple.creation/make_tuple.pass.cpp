//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class... Types>
//   tuple<VTypes...> make_tuple(Types&&... t);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/tuple>

#include "test_macros.h"

int main(int, char**)
{
  {
    int i                                 = 0;
    float j                               = 0;
    cuda::std::tuple<int, int&, float&> t = cuda::std::make_tuple(1, cuda::std::ref(i), cuda::std::ref(j));
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 0);
    assert(cuda::std::get<2>(t) == 0);
    i = 2;
    j = 3.5;
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
    assert(cuda::std::get<2>(t) == 3.5);
    cuda::std::get<1>(t) = 0;
    cuda::std::get<2>(t) = 0;
    assert(i == 0);
    assert(j == 0);
  }
#if TEST_STD_VER > 2011
  {
    constexpr auto t1   = cuda::std::make_tuple(0, 1, 3.14);
    constexpr int i1    = cuda::std::get<1>(t1);
    constexpr double d1 = cuda::std::get<2>(t1);
    static_assert(i1 == 1, "");
    static_assert(d1 == 3.14, "");
  }
#endif

  return 0;
}
