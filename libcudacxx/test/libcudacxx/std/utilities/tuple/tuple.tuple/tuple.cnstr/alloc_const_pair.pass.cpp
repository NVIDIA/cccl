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

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    cuda::std::pair<long, int> p{2, 3};
    cuda::std::tuple<long long, double> t(cuda::std::allocator_arg, alloc, p);
    assert(cuda::std::get<0>(t) == 2);
    assert(cuda::std::get<1>(t) == 3);
  }
  {
    cuda::std::pair<int, int> p{3, 4};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, p);
    assert(cuda::std::get<0>(t).value == 3);
    assert(cuda::std::get<1>(t).value == 4);
  }
  {
    const cuda::std::pair<int, int> p{5, 6};
    cuda::std::tuple<constexpr_alloc_last, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, p);
    assert(cuda::std::get<0>(t).value == 5);
    assert(cuda::std::get<1>(t).value == 6);
  }
  {
    cuda::std::pair<int, int> p{2, 3};
    cuda::std::tuple<constexpr_alloc_arg, double> t(cuda::std::allocator_arg, alloc, p);
    assert(cuda::std::get<0>(t).value == 2);
    assert(cuda::std::get<1>(t) == 3);
  }
  {
    cuda::std::pair<int, int> p{2, 3};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, p);
    assert(cuda::std::get<0>(t).value == 2);
    assert(cuda::std::get<1>(t).value == 3);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
