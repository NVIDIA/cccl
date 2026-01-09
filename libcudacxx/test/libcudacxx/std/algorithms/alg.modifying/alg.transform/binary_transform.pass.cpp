//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter1, InputIterator InIter2, class OutIter,
//          Callable<auto, const InIter1::value_type&, const InIter2::value_type&> BinaryOp>
//   requires OutputIterator<OutIter, BinaryOp::result_type> && CopyConstructible<BinaryOp>
// constexpr OutIter      // constexpr after C++17
// transform(InIter1 first1, InIter1 last1, InIter2 first2, OutIter result, BinaryOp binary_op);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

template <class InIter1, class InIter2, class OutIter>
constexpr __host__ __device__ void test()
{
  {
    constexpr int N           = 5;
    constexpr int ia[N]       = {0, 1, 2, 3, 4};
    constexpr int ib[N]       = {1, 2, 3, 4, 5};
    constexpr int expected[N] = {1, 3, 5, 7, 9};
    int ic[N + 1]             = {0, 0, 0, 0, 0, 0};

    OutIter r = cuda::std::transform(InIter1(ia), InIter1(ia + N), InIter2(ib), OutIter(ic), cuda::std::plus<int>{});
    assert(base(r) == ic + N);
    for (int i = 0; i < N; ++i)
    {
      assert(ic[i] == expected[i]);
    }

    for (unsigned i = N; i < N + 1; ++i)
    {
      assert(ic[i] == 0);
    }
  }
}

constexpr __host__ __device__ bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>, int*>();

  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>, int*>();

  test<cpp17_input_iterator<const int*>, const int*, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, const int*, cpp17_input_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, const int*, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, const int*, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, const int*, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, const int*, int*>();

  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, const int*, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, const int*, cpp17_input_iterator<int*>>();
  test<forward_iterator<const int*>, const int*, forward_iterator<int*>>();
  test<forward_iterator<const int*>, const int*, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, const int*, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, const int*, int*>();

  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>, int*>();

  test<bidirectional_iterator<const int*>, const int*, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, const int*, cpp17_input_iterator<int*>>();
  test<bidirectional_iterator<const int*>, const int*, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, const int*, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, const int*, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, const int*, int*>();

  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, const int*, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, const int*, cpp17_input_iterator<int*>>();
  test<random_access_iterator<const int*>, const int*, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, const int*, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, const int*, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, const int*, int*>();

  test<const int*, cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<const int*, cpp17_input_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<const int*, cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<const int*, cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<const int*, cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<const int*, cpp17_input_iterator<const int*>, int*>();

  test<const int*, forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<const int*, forward_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<const int*, forward_iterator<const int*>, forward_iterator<int*>>();
  test<const int*, forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<const int*, forward_iterator<const int*>, random_access_iterator<int*>>();
  test<const int*, forward_iterator<const int*>, int*>();

  test<const int*, bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<const int*, bidirectional_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<const int*, bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<const int*, bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<const int*, bidirectional_iterator<const int*>, int*>();

  test<const int*, random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<const int*, random_access_iterator<const int*>, cpp17_input_iterator<int*>>();
  test<const int*, random_access_iterator<const int*>, forward_iterator<int*>>();
  test<const int*, random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<const int*, random_access_iterator<const int*>, int*>();

  test<const int*, const int*, cpp17_output_iterator<int*>>();
  test<const int*, const int*, cpp17_input_iterator<int*>>();
  test<const int*, const int*, forward_iterator<int*>>();
  test<const int*, const int*, bidirectional_iterator<int*>>();
  test<const int*, const int*, random_access_iterator<int*>>();
  test<const int*, const int*, int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
