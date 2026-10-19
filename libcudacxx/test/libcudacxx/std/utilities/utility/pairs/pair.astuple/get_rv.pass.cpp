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
//     typename tuple_element<I, cuda::std::pair<T1, T2> >::type&&
//     get(pair<T1, T2>&&);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
#ifndef CCCL_FORCE_TILE_TESTS
  {
    using P = cuda::std::pair<cuda::std::unique_ptr<int>, short>;
    P p(cuda::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
    cuda::std::unique_ptr<int> ptr = cuda::std::get<0>(cuda::std::move(p));
    assert(*ptr == 3);
  }
#endif // CCCL_FORCE_TILE_TESTS

  {
    using P = cuda::std::pair<MoveOnly, short>;
    P p(3, static_cast<short>(4));
    assert(cuda::std::get<0>(cuda::std::move(p)) == 3);
  }

  return 0;
}
