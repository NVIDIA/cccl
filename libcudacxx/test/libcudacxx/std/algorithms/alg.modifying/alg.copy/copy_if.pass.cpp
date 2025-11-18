//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter,
//          Predicate<auto, InIter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr OutIter   // constexpr after C++17
//   copy_if(InIter first, InIter last, OutIter result, Pred pred);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct PredMod3
{
  __host__ __device__ constexpr bool operator()(int i) const noexcept
  {
    return i % 3 == 0;
  }
};

struct PredEqual6
{
  __host__ __device__ constexpr bool operator()(int i) const noexcept
  {
    return i == 6;
  }
};

template <class InIter, class OutIter>
constexpr __host__ __device__ void test()
{
  {
    constexpr unsigned N = 1000;
    int ia[N]            = {0};
    for (unsigned i = 0; i < N; ++i)
    {
      ia[i] = i;
    }
    int ib[N] = {0};

    OutIter r = cuda::std::copy_if(InIter(ia), InIter(ia + N), OutIter(ib), PredMod3{});
    assert(base(r) == ib + N / 3 + 1);
    for (unsigned i = 0; i < N / 3 + 1; ++i)
    {
      assert(ib[i] % 3 == 0);
    }
  }
  {
    constexpr int N               = 5;
    constexpr int expected_copies = 2;
    int ia[N]                     = {2, 4, 6, 8, 6};
    int ic[N + 2]                 = {0, 0, 0, 0, 0, 0};

    auto p = cuda::std::copy_if(ia, ia + N, ic, PredEqual6{});
    assert(p == (ic + expected_copies));
    for (unsigned i = 0; i < expected_copies; ++i)
    {
      assert(ic[i] == 6);
    }

    for (unsigned i = expected_copies; i < N + 2; ++i)
    {
      assert(ic[i] == 0);
    }
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  test<const int*, cpp17_output_iterator<int*>>();
  test<const int*, cpp17_input_iterator<int*>>();
  test<const int*, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<int*>>();
  test<const int*, int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
