//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter   // constexpr after C++17
//   copy_n(InIter first, InIter::difference_type n, OutIter result);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "user_defined_integral.h"

using UDI = UserDefinedIntegral<unsigned>;

template <class InIter, class OutIter>
TEST_CONSTEXPR_CXX20 __host__ __device__ void test()
{
  {
    constexpr unsigned N = 1000;
    int ia[N];
    for (unsigned i = 0; i < N; ++i)
    {
      ia[i] = i;
    }
    int ib[N] = {0};

    OutIter r = cuda::std::copy_n(InIter(ia), UDI(N / 2), OutIter(ib));
    assert(base(r) == ib + N / 2);
    for (unsigned i = 0; i < N / 2; ++i)
    {
      assert(ia[i] == ib[i]);
    }
  }
  {
    constexpr int N = 5;
    int ia[N]       = {1, 2, 3, 4, 5};
    int ic[N + 2]   = {6, 6, 6, 6, 6, 6, 6};

    auto p = cuda::std::copy_n(ia, N, ic);
    assert(p == (ic + N));
    for (unsigned i = 0; i < N; ++i)
    {
      assert(ia[i] == ic[i]);
    }

    for (unsigned i = N; i < N + 2; ++i)
    {
      assert(ic[i] == 6);
    }
  }
}

TEST_CONSTEXPR_CXX20 __host__ __device__ bool test()
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

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
