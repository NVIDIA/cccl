//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, const tuple<UTypes...>&);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

struct Explicit
{
  int value;
  __host__ __device__ explicit Explicit(int x)
      : value(x)
  {}
};

struct Implicit
{
  int value;
  __host__ __device__ Implicit(int x)
      : value(x)
  {}
};

int main(int, char**)
{
  alloc_first::allocator_constructed() = false;
  alloc_last::allocator_constructed()  = false;

  {
    typedef cuda::std::tuple<long> T0;
    typedef cuda::std::tuple<long long> T1;
    T0 t0(2);
    T1 t1(cuda::std::allocator_arg, A1<int>(), t0);
    assert(cuda::std::get<0>(t1) == 2);
  }
  {
    typedef cuda::std::tuple<int> T0;
    typedef cuda::std::tuple<alloc_first> T1;
    T0 t0(2);
    alloc_first::allocator_constructed() = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 2);
  }
  {
    typedef cuda::std::tuple<int, int> T0;
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
  {
    typedef cuda::std::tuple<long, int, int> T0;
    typedef cuda::std::tuple<long long, alloc_first, alloc_last> T1;
    T0 t0(1, 2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 1);
    assert(cuda::std::get<1>(t1) == 2);
    assert(cuda::std::get<2>(t1) == 3);
  }
  // cuda::std::allocator is unsupported
  /*
  {
      const cuda::std::tuple<int> t1(42);
      cuda::std::tuple<Explicit> t2{cuda::std::allocator_arg, cuda::std::allocator<void>{},  t1};
      assert(cuda::std::get<0>(t2).value == 42);
  }
  {
      const cuda::std::tuple<int> t1(42);
      cuda::std::tuple<Implicit> t2 = {cuda::std::allocator_arg, cuda::std::allocator<void>{}, t1};
      assert(cuda::std::get<0>(t2).value == 42);
  }
  */

  return 0;
}
