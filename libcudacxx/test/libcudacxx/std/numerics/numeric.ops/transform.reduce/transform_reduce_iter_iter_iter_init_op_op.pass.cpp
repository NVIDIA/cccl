//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <class InputIterator1, class InputIterator2, class T,
//           class BinaryOperation1, class BinaryOperation2>
//    T transform_reduce(InputIterator1 first1, InputIterator1 last1,
//                       InputIterator2 first2, T init,
//                       BinaryOperation1 binary_op1, BinaryOperation2 binary_op2);
//

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/numeric>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class Iter2, class T, class Op1, class Op2>
__host__ __device__ TEST_CONSTEXPR_CXX14 void
test(Iter1 first1, Iter1 last1, Iter2 first2, T init, Op1 op1, Op2 op2, T x)
{
  static_assert(
    cuda::std::is_same<T, decltype(cuda::std::transform_reduce(first1, last1, first2, init, op1, op2))>::value, "");
  assert(cuda::std::transform_reduce(first1, last1, first2, init, op1, op2) == x);
}

template <class SIter, class UIter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]          = {1, 2, 3, 4, 5, 6};
  unsigned int ua[] = {2, 4, 6, 8, 10, 12};
  unsigned sa       = sizeof(ia) / sizeof(ia[0]);
  assert(sa == sizeof(ua) / sizeof(ua[0])); // just to be sure

  test(SIter(ia), SIter(ia), UIter(ua), 0, cuda::std::plus<>(), cuda::std::multiplies<>(), 0);
  test(UIter(ua), UIter(ua), SIter(ia), 1, cuda::std::multiplies<>(), cuda::std::plus<>(), 1);
  test(SIter(ia), SIter(ia + 1), UIter(ua), 0, cuda::std::multiplies<>(), cuda::std::plus<>(), 0);
  test(UIter(ua), UIter(ua + 1), SIter(ia), 2, cuda::std::plus<>(), cuda::std::multiplies<>(), 4);
  test(SIter(ia), SIter(ia + 2), UIter(ua), 0, cuda::std::plus<>(), cuda::std::multiplies<>(), 10);
  test(UIter(ua), UIter(ua + 2), SIter(ia), 3, cuda::std::multiplies<>(), cuda::std::plus<>(), 54);
  test(SIter(ia), SIter(ia + sa), UIter(ua), 4, cuda::std::multiplies<>(), cuda::std::plus<>(), 2099520);
  test(UIter(ua), UIter(ua + sa), SIter(ia), 4, cuda::std::plus<>(), cuda::std::multiplies<>(), 186);
}

template <typename T, typename Init>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_return_type()
{
  T* p = nullptr;
  unused(p);
  static_assert(
    cuda::std::is_same<
      Init,
      decltype(cuda::std::transform_reduce(p, p, p, Init{}, cuda::std::plus<>(), cuda::std::multiplies<>()))>::value,
    "");
}

struct SumMoveOnly
{
  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator()(const MoveOnly& lhs, const MoveOnly& rhs) const noexcept
  {
    return MoveOnly{lhs.get() + rhs.get()};
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_move_only_types()
{
  MoveOnly ia[] = {{1}, {2}, {3}};
  MoveOnly ib[] = {{1}, {2}, {3}};
  assert(12
         == cuda::std::transform_reduce(
              cuda::std::begin(ia), cuda::std::end(ia), cuda::std::begin(ib), MoveOnly{0}, SumMoveOnly{}, SumMoveOnly{})
              .get());
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test_return_type<char, int>();
  test_return_type<int, int>();
  test_return_type<int, unsigned long>();
  test_return_type<float, int>();
  test_return_type<short, float>();
  test_return_type<double, char>();
  test_return_type<char, double>();

  //  All the iterator categories
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const unsigned int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const unsigned int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const unsigned int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const unsigned int*>>();

  test<forward_iterator<const int*>, cpp17_input_iterator<const unsigned int*>>();
  test<forward_iterator<const int*>, forward_iterator<const unsigned int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const unsigned int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const unsigned int*>>();

  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const unsigned int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const unsigned int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const unsigned int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const unsigned int*>>();

  test<random_access_iterator<const int*>, cpp17_input_iterator<const unsigned int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const unsigned int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const unsigned int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const unsigned int*>>();

  //  just plain pointers (const vs. non-const, too)
  test<const int*, const unsigned int*>();
  test<const int*, unsigned int*>();
  test<int*, const unsigned int*>();
  test<int*, unsigned int*>();

  test_move_only_types();

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
