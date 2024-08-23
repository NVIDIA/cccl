//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator InIter, OutputIterator<auto, InIter::reference> OutIter>
//   constexpr OutIter          // constexpr after C++17
//   rotate_copy(InIter first, InIter middle, InIter last, OutIter result);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

template <class InIter, class OutIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]          = {0, 1, 2, 3};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  int ib[sa]        = {0};

  OutIter r = cuda::std::rotate_copy(InIter(ia), InIter(ia), InIter(ia), OutIter(ib));
  assert(base(r) == ib);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia), InIter(ia + 1), OutIter(ib));
  assert(base(r) == ib + 1);
  assert(ib[0] == 0);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 1), InIter(ia + 1), OutIter(ib));
  assert(base(r) == ib + 1);
  assert(ib[0] == 0);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia), InIter(ia + 2), OutIter(ib));
  assert(base(r) == ib + 2);
  assert(ib[0] == 0);
  assert(ib[1] == 1);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 1), InIter(ia + 2), OutIter(ib));
  assert(base(r) == ib + 2);
  assert(ib[0] == 1);
  assert(ib[1] == 0);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 2), InIter(ia + 2), OutIter(ib));
  assert(base(r) == ib + 2);
  assert(ib[0] == 0);
  assert(ib[1] == 1);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia), InIter(ia + 3), OutIter(ib));
  assert(base(r) == ib + 3);
  assert(ib[0] == 0);
  assert(ib[1] == 1);
  assert(ib[2] == 2);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 1), InIter(ia + 3), OutIter(ib));
  assert(base(r) == ib + 3);
  assert(ib[0] == 1);
  assert(ib[1] == 2);
  assert(ib[2] == 0);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 2), InIter(ia + 3), OutIter(ib));
  assert(base(r) == ib + 3);
  assert(ib[0] == 2);
  assert(ib[1] == 0);
  assert(ib[2] == 1);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 3), InIter(ia + 3), OutIter(ib));
  assert(base(r) == ib + 3);
  assert(ib[0] == 0);
  assert(ib[1] == 1);
  assert(ib[2] == 2);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia), InIter(ia + 4), OutIter(ib));
  assert(base(r) == ib + 4);
  assert(ib[0] == 0);
  assert(ib[1] == 1);
  assert(ib[2] == 2);
  assert(ib[3] == 3);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 1), InIter(ia + 4), OutIter(ib));
  assert(base(r) == ib + 4);
  assert(ib[0] == 1);
  assert(ib[1] == 2);
  assert(ib[2] == 3);
  assert(ib[3] == 0);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 2), InIter(ia + 4), OutIter(ib));
  assert(base(r) == ib + 4);
  assert(ib[0] == 2);
  assert(ib[1] == 3);
  assert(ib[2] == 0);
  assert(ib[3] == 1);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 3), InIter(ia + 4), OutIter(ib));
  assert(base(r) == ib + 4);
  assert(ib[0] == 3);
  assert(ib[1] == 0);
  assert(ib[2] == 1);
  assert(ib[3] == 2);

  r = cuda::std::rotate_copy(InIter(ia), InIter(ia + 4), InIter(ia + 4), OutIter(ib));
  assert(base(r) == ib + 4);
  assert(ib[0] == 0);
  assert(ib[1] == 1);
  assert(ib[2] == 2);
  assert(ib[3] == 3);

  {
    int ints[]        = {1, 3, 5, 2, 5, 6};
    int const n_ints  = sizeof(ints) / sizeof(int);
    int zeros[n_ints] = {0};

    const cuda::std::size_t N = 2;
    const auto middle         = cuda::std::begin(ints) + N;
    auto it = cuda::std::rotate_copy(cuda::std::begin(ints), middle, cuda::std::end(ints), cuda::std::begin(zeros));
    assert(cuda::std::distance(cuda::std::begin(zeros), it) == n_ints);
    assert(cuda::std::equal(cuda::std::begin(ints), middle, cuda::std::begin(zeros) + n_ints - N));
    assert(cuda::std::equal(middle, cuda::std::end(ints), cuda::std::begin(zeros)));
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<bidirectional_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<int*>>();
  test<bidirectional_iterator<const int*>, int*>();

  test<random_access_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<random_access_iterator<const int*>, forward_iterator<int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<int*>>();
  test<random_access_iterator<const int*>, int*>();

  test<const int*, cpp17_output_iterator<int*>>();
  test<const int*, forward_iterator<int*>>();
  test<const int*, bidirectional_iterator<int*>>();
  test<const int*, random_access_iterator<int*>>();
  test<const int*, int*>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
  return 0;
}
