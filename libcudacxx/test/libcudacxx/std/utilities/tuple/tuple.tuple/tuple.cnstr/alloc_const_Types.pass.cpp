//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, const Types&...);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

struct ImplicitCopy
{
  __host__ __device__ explicit ImplicitCopy(int) {}
  __host__ __device__ ImplicitCopy(ImplicitCopy const&) {}
};

// cuda::std::allocator not supported
/*
// Test that tuple(cuda::std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
cuda::std::tuple<ImplicitCopy> testImplicitCopy1() {
    ImplicitCopy i(42);
    return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, i};
}

cuda::std::tuple<ImplicitCopy> testImplicitCopy2() {
    const ImplicitCopy i(42);
    return {cuda::std::allocator_arg, cuda::std::allocator<void>{}, i};
}
*/

int main(int, char**)
{
  // Static initialization not supported on GPUs
  alloc_first::allocator_constructed() = false;
  alloc_last::allocator_constructed()  = false;
  // cuda::std::allocator not supported
  /*
  {
      // check that the literal '0' can implicitly initialize a stored pointer.
      cuda::std::tuple<int*> t = {cuda::std::allocator_arg, cuda::std::allocator<void>{}, 0};
  }
  */
  {
    cuda::std::tuple<int> t(cuda::std::allocator_arg, A1<int>(), 3);
    assert(cuda::std::get<0>(t) == 3);
  }
  {
    assert(!alloc_first::allocator_constructed());
    cuda::std::tuple<alloc_first> t(cuda::std::allocator_arg, A1<int>(5), alloc_first(3));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == alloc_first(3));
  }
  {
    assert(!alloc_last::allocator_constructed());
    cuda::std::tuple<alloc_last> t(cuda::std::allocator_arg, A1<int>(5), alloc_last(3));
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == alloc_last(3));
  }
  {
    alloc_first::allocator_constructed() = false;
    cuda::std::tuple<int, alloc_first> t(cuda::std::allocator_arg, A1<int>(5), 10, alloc_first(15));
    assert(cuda::std::get<0>(t) == 10);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first(15));
  }
  {
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    cuda::std::tuple<int, alloc_first, alloc_last> t(
      cuda::std::allocator_arg, A1<int>(5), 1, alloc_first(2), alloc_last(3));
    assert(cuda::std::get<0>(t) == 1);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first(2));
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<2>(t) == alloc_last(3));
  }
  {
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    cuda::std::tuple<int, alloc_first, alloc_last> t(
      cuda::std::allocator_arg, A2<int>(5), 1, alloc_first(2), alloc_last(3));
    assert(cuda::std::get<0>(t) == 1);
    assert(!alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first(2));
    assert(!alloc_last::allocator_constructed());
    assert(cuda::std::get<2>(t) == alloc_last(3));
  }

  return 0;
}
