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
//   tuple(allocator_arg_t, const Alloc& a, const tuple&);

// UNSUPPORTED: c++98, c++03

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::tuple<> T;
    T t0;
    T t(cuda::std::allocator_arg, A1<int>(), t0);
  }
  {
    typedef cuda::std::tuple<int> T;
    T t0(2);
    T t(cuda::std::allocator_arg, A1<int>(), t0);
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    typedef cuda::std::tuple<alloc_first> T;
    T t0(2);
    alloc_first::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    typedef cuda::std::tuple<alloc_last> T;
    T t0(2);
    alloc_last::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
  }
// testing extensions
#ifdef _LIBCUDACXX_VERSION
  {
    typedef cuda::std::tuple<alloc_first, alloc_last> T;
    T t0(2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 3);
  }
  {
    typedef cuda::std::tuple<int, alloc_first, alloc_last> T;
    T t0(1, 2, 3);
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t) == 2);
    assert(cuda::std::get<2>(t) == 3);
  }
#endif

  return 0;
}
