//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class U1, class U2>
//   tuple(allocator_arg_t, const Alloc& a, const pair<U1, U2>&);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<long, int> T0;
    typedef cuda::std::tuple<long long, double> T1;
    T0 t0(2, 3);
    T1 t1(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == 3);
  }
  {
    typedef cuda::std::pair<int, int> T0;
    typedef cuda::std::tuple<alloc_first, double> T1;
    T0 t0(2, 3);
    alloc_first::allocator_constructed() = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == 3);
  }
  {
    typedef cuda::std::pair<int, int> T0;
    typedef cuda::std::tuple<alloc_first, alloc_last> T1;
    T0 t0(2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == 3);
  }

  return 0;
}
