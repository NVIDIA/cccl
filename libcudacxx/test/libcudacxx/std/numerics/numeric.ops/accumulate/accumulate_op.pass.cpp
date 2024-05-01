//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <InputIterator Iter, MoveConstructible T,
//           Callable<auto, const T&, Iter::reference> BinaryOperation>
//   requires HasAssign<T, BinaryOperation::result_type>
//         && CopyConstructible<BinaryOperation>
//   T
//   accumulate(Iter first, Iter last, T init, BinaryOperation binary_op);

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/numeric>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING

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
  rvalue_addable arr[100];
  auto res1 = cuda::std::accumulate(arr, arr + 100, rvalue_addable());
  auto res2 = cuda::std::accumulate(arr, arr + 100, rvalue_addable(), /*predicate=*/rvalue_addable());
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

template <class Iter, class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(Iter first, Iter last, T init, T x)
{
  assert(cuda::std::accumulate(first, last, init, cuda::std::multiplies<T>()) == x);
}

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);
  test(Iter(ia), Iter(ia), 1, 1);
  test(Iter(ia), Iter(ia), 10, 10);
  test(Iter(ia), Iter(ia + 1), 1, 1);
  test(Iter(ia), Iter(ia + 1), 10, 10);
  test(Iter(ia), Iter(ia + 2), 1, 2);
  test(Iter(ia), Iter(ia + 2), 10, 20);
  test(Iter(ia), Iter(ia + sa), 1, 720);
  test(Iter(ia), Iter(ia + sa), 10, 7200);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

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
