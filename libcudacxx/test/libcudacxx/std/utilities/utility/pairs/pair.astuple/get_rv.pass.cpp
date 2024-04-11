//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, cuda::std::pair<T1, T2> >::type&&
//     get(pair<T1, T2>&&);

#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  // cuda/std/memory not supported
  /*
  {
      typedef cuda::std::pair<cuda::std::unique_ptr<int>, short> P;
      P p(cuda::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
      cuda::std::unique_ptr<int> ptr = cuda::std::get<0>(cuda::std::move(p));
      assert(*ptr == 3);
  }
  */
  return 0;
}
