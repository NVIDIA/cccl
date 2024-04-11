//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, cuda::std::pair<T1, T2> >::type&
//     get(pair<T1, T2>&);

// UNSUPPORTED: msvc

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

#if TEST_STD_VER > 2011
struct S
{
  cuda::std::pair<int, int> a;
  int k;
  __device__ __host__ constexpr S()
      : a{1, 2}
      , k(cuda::std::get<0>(a))
  {}
};

__device__ __host__ constexpr cuda::std::pair<int, int> getP()
{
  return {3, 4};
}
#endif

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P;
    P p(3, static_cast<short>(4));
    assert(cuda::std::get<0>(p) == 3);
    assert(cuda::std::get<1>(p) == 4);
    cuda::std::get<0>(p) = 5;
    cuda::std::get<1>(p) = 6;
    assert(cuda::std::get<0>(p) == 5);
    assert(cuda::std::get<1>(p) == 6);
  }

#if TEST_STD_VER > 2011
  {
    static_assert(S().k == 1, "");
    static_assert(cuda::std::get<1>(getP()) == 4, "");
  }
#endif

  return 0;
}
