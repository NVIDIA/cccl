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
//     const typename tuple_element<I, cuda::std::pair<T1, T2> >::type&
//     get(const pair<T1, T2>&);

#include <cuda/std/cassert>
#include <cuda/std/utility>

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P;
    const P p(3, 4);
    assert(cuda::std::get<0>(p) == 3);
    assert(cuda::std::get<1>(p) == 4);
    cuda::std::get<0>(p) = 5;
  }

  return 0;
}
