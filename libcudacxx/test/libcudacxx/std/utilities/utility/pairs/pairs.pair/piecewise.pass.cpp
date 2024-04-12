//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

// <utility>

// template <class T1, class T2> struct pair

// template <class... Args1, class... Args2>
//     pair(piecewise_construct_t, tuple<Args1...> first_args,
//                                 tuple<Args2...> second_args);

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, int*> P1;
    typedef cuda::std::pair<int*, int> P2;
    typedef cuda::std::pair<P1, P2> P3;
    P3 p3(
      cuda::std::piecewise_construct, cuda::std::tuple<int, int*>(3, nullptr), cuda::std::tuple<int*, int>(nullptr, 4));
    assert(p3.first == P1(3, nullptr));
    assert(p3.second == P2(nullptr, 4));
  }

  return 0;
}
