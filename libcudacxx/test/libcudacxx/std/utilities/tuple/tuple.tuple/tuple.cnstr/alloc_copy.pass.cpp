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

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    using T = cuda::std::tuple<>;
    T t0;
    T t(cuda::std::allocator_arg, A1<int>(), t0);
  }
  {
    using T = cuda::std::tuple<int>;
    T t0(2);
    T t(cuda::std::allocator_arg, A1<int>(), t0);
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    using T = cuda::std::tuple<alloc_first>;
    T t0(2);
    alloc_first::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    using T = cuda::std::tuple<alloc_last>;
    T t0(2);
    alloc_last::allocator_constructed() = false;
    T t(cuda::std::allocator_arg, A1<int>(5), t0);
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == 2);
  }
// testing extensions
#ifdef _CUDA_STD_VERSION
  {
    using T = cuda::std::tuple<alloc_first, alloc_last>;
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
    using T = cuda::std::tuple<int, alloc_first, alloc_last>;
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
