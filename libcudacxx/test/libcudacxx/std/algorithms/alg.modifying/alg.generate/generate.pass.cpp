//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, Callable Generator>
//   requires OutputIterator<Iter, Generator::result_type>
//         && CopyConstructible<Generator>
//   constexpr void      // constexpr after c++17
//   generate(Iter first, Iter last, Generator gen);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct gen_test
{
  constexpr __host__ __device__ int operator()() const noexcept
  {
    return 1;
  }
};

template <class Iter>
constexpr __host__ __device__ void test()
{
  constexpr int N = 5;
  int ia[N + 1]   = {0};
  cuda::std::generate(Iter(ia), Iter(ia + N), gen_test());
  for (int i = 0; i < N; ++i)
  {
    assert(ia[i] == 1);
  }

  for (int i = N; i < N + 1; ++i)
  {
    assert(ia[i] == 0);
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
