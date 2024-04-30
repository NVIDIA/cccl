//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator Iter1, InputIterator Iter2, MoveConstructible T,
//           class BinaryOperation1,
//           Callable<auto, Iter1::reference, Iter2::reference> BinaryOperation2>
//   requires Callable<BinaryOperation1, const T&, BinaryOperation2::result_type>
//         && HasAssign<T, BinaryOperation1::result_type>
//         && CopyConstructible<BinaryOperation1>
//         && CopyConstructible<BinaryOperation2>
//   T
//   inner_product(Iter1 first1, Iter1 last1, Iter2 first2,
//                 T init, BinaryOperation1 binary_op1, BinaryOperation2 binary_op2);

#include <cuda/std/functional>
#include <cuda/std/numeric>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

struct do_nothing_op
{
  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX14 T operator()(T a, T)
  {
    return a;
  }
};

struct rvalue_addable
{
  bool correctOperatorUsed = false;

  __host__ __device__ TEST_CONSTEXPR_CXX14 rvalue_addable operator*(rvalue_addable const&)
  {
    return *this;
  }

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
  rvalue_addable arr[100];
  auto res1 = cuda::std::inner_product(arr, arr + 100, arr, rvalue_addable());
  auto res2 =
    cuda::std::inner_product(arr, arr + 100, arr, rvalue_addable(), /*predicate=*/rvalue_addable(), do_nothing_op());

  assert(res1.correctOperatorUsed);
  assert(res2.correctOperatorUsed);
}

#ifdef _LIBCUDACXX_HAS_STRING
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_string()
{
  cuda::std::string sa[] = {"a", "b", "c"};
  assert(cuda::std::accumulate(sa, sa + 3, cuda::std::string()) == "abc");
  assert(cuda::std::accumulate(sa, sa + 3, cuda::std::string(), cuda::std::plus<cuda::std::string>()) == "abc");
}
#endif // _LIBCUDACXX_HAS_STRING

template <class Iter1, class Iter2, class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(Iter1 first1, Iter1 last1, Iter2 first2, T init, T x)
{
  assert(cuda::std::inner_product(first1, last1, first2, init, cuda::std::multiplies<int>(), cuda::std::plus<int>())
         == x);
}

template <class Iter1, class Iter2>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int a[]     = {1, 2, 3, 4, 5, 6};
  int b[]     = {6, 5, 4, 3, 2, 1};
  unsigned sa = sizeof(a) / sizeof(a[0]);
  test(Iter1(a), Iter1(a), Iter2(b), 1, 1);
  test(Iter1(a), Iter1(a), Iter2(b), 10, 10);
  test(Iter1(a), Iter1(a + 1), Iter2(b), 1, 7);
  test(Iter1(a), Iter1(a + 1), Iter2(b), 10, 70);
  test(Iter1(a), Iter1(a + 2), Iter2(b), 1, 49);
  test(Iter1(a), Iter1(a + 2), Iter2(b), 10, 490);
  test(Iter1(a), Iter1(a + sa), Iter2(b), 1, 117649);
  test(Iter1(a), Iter1(a + sa), Iter2(b), 10, 1176490);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<cpp17_input_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, forward_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, random_access_iterator<const int*>>();
  test<cpp17_input_iterator<const int*>, const int*>();

  test<forward_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>, forward_iterator<const int*>>();
  test<forward_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<forward_iterator<const int*>, random_access_iterator<const int*>>();
  test<forward_iterator<const int*>, const int*>();

  test<bidirectional_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, random_access_iterator<const int*>>();
  test<bidirectional_iterator<const int*>, const int*>();

  test<random_access_iterator<const int*>, cpp17_input_iterator<const int*>>();
  test<random_access_iterator<const int*>, forward_iterator<const int*>>();
  test<random_access_iterator<const int*>, bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>, random_access_iterator<const int*>>();
  test<random_access_iterator<const int*>, const int*>();

  test<const int*, cpp17_input_iterator<const int*>>();
  test<const int*, forward_iterator<const int*>>();
  test<const int*, bidirectional_iterator<const int*>>();
  test<const int*, random_access_iterator<const int*>>();
  test<const int*, const int*>();

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
