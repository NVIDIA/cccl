//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template<InputIterator InIter,
//          OutputIterator<auto, const InIter::value_type&> OutIter,
//          Callable<auto, const InIter::value_type&, InIter::reference> BinaryOperation>
//   requires HasAssign<InIter::value_type, BinaryOperation::result_type>
//         && Constructible<InIter::value_type, InIter::reference>
//         && CopyConstructible<BinaryOperation>
//   OutIter
//   partial_sum(InIter first, InIter last, OutIter result, BinaryOperation binary_op);

#include <cuda/std/functional>
#include <cuda/std/numeric>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct rvalue_addable
{
  bool correctOperatorUsed = false;

  // make sure the predicate is passed an rvalue and an lvalue (so check that the first argument was moved)
  __host__ __device__ TEST_CONSTEXPR_CXX14 rvalue_addable operator()(rvalue_addable&& r, rvalue_addable const&)
  {
    r.correctOperatorUsed = true;
    return cuda::std::move(r);
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX14 rvalue_addable operator+(rvalue_addable& lhs, rvalue_addable const&)
{
  lhs.correctOperatorUsed = false;
  return lhs;
}

__host__ __device__ TEST_CONSTEXPR_CXX14 rvalue_addable operator+(rvalue_addable&& lhs, rvalue_addable const&)
{
  lhs.correctOperatorUsed = true;
  return cuda::std::move(lhs);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_use_move()
{
  const cuda::std::size_t size = 100;
  rvalue_addable arr[size];
  rvalue_addable res1[size];
  rvalue_addable res2[size];
  cuda::std::partial_sum(arr, arr + size, res1);
  cuda::std::partial_sum(arr, arr + size, res2, /*predicate=*/rvalue_addable());
  // start at 1 because the first element is not moved
  for (unsigned i = 1; i < size; ++i)
  {
    assert(res1[i].correctOperatorUsed);
  }
  for (unsigned i = 1; i < size; ++i)
  {
    assert(res2[i].correctOperatorUsed);
  }
}

#ifdef _LIBCUDACXX_HAS_STRING
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_string()
{
  cuda::std::string sa[] = {"a", "b", "c"};
  cuda::std::string sr[] = {"a", "ba", "cb"};
  cuda::std::string output[3];
  cuda::std::adjacent_difference(sa, sa + 3, output, cuda::std::plus<cuda::std::string>());
  for (unsigned i = 0; i < 3; ++i)
  {
    assert(output[i] == sr[i]);
  }
}
#endif // _LIBCUDACXX_HAS_STRING

template <class InIter, class OutIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]         = {1, 2, 3, 4, 5};
  int ir[]         = {1, -1, -4, -8, -13};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  int ib[s]        = {0};
  OutIter r        = cuda::std::partial_sum(InIter(ia), InIter(ia + s), OutIter(ib), cuda::std::minus<int>());
  assert(base(r) == ib + s);
  for (unsigned i = 0; i < s; ++i)
  {
    assert(ib[i] == ir[i]);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<int*>>();
  test<cpp17_input_iterator<const int*>, int*>();

  test<forward_iterator<const int*>, cpp17_output_iterator<int*>>();
  test<forward_iterator<const int*>, forward_iterator<int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<int*>>();
  test<forward_iterator<const int*>, random_access_iterator<int*>>();
  test<forward_iterator<const int*>, int*>();

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

  test_use_move();

#ifdef _LIBCUDACXX_HAS_STRING
  test_string();
#endif // _LIBCUDACXX_HAS_STRING

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014

  return 0;
}
