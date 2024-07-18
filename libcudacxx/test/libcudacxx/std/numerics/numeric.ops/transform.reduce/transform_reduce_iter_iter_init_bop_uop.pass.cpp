//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template <class InputIterator1, class T,
//           class BinaryOperation, class UnaryOperation>
//    T transform_reduce(InputIterator1 first1, InputIterator1 last1,
//                       T init, BinaryOperation binary_op, UnaryOperation unary_op);
//

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/numeric>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

struct identity
{
  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX14 T operator()(T&& x) const
  {
    return cuda::std::forward<T>(x);
  }
};

struct twice
{
  template <class T>
  __host__ __device__ TEST_CONSTEXPR_CXX14 T operator()(const T& x) const
  {
    return 2 * x;
  }
};

template <class Iter1, class T, class BOp, class UOp>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test(Iter1 first1, Iter1 last1, T init, BOp bOp, UOp uOp, T x)
{
  static_assert(cuda::std::is_same<T, decltype(cuda::std::transform_reduce(first1, last1, init, bOp, uOp))>::value, "");
  assert(cuda::std::transform_reduce(first1, last1, init, bOp, uOp) == x);
}

template <class Iter>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  int ia[]    = {1, 2, 3, 4, 5, 6};
  unsigned sa = sizeof(ia) / sizeof(ia[0]);

  test(Iter(ia), Iter(ia), 0, cuda::std::plus<>(), identity(), 0);
  test(Iter(ia), Iter(ia), 1, cuda::std::multiplies<>(), identity(), 1);
  test(Iter(ia), Iter(ia + 1), 0, cuda::std::multiplies<>(), identity(), 0);
  test(Iter(ia), Iter(ia + 1), 2, cuda::std::plus<>(), identity(), 3);
  test(Iter(ia), Iter(ia + 2), 0, cuda::std::plus<>(), identity(), 3);
  test(Iter(ia), Iter(ia + 2), 3, cuda::std::multiplies<>(), identity(), 6);
  test(Iter(ia), Iter(ia + sa), 4, cuda::std::multiplies<>(), identity(), 2880);
  test(Iter(ia), Iter(ia + sa), 4, cuda::std::plus<>(), identity(), 25);

  test(Iter(ia), Iter(ia), 0, cuda::std::plus<>(), twice(), 0);
  test(Iter(ia), Iter(ia), 1, cuda::std::multiplies<>(), twice(), 1);
  test(Iter(ia), Iter(ia + 1), 0, cuda::std::multiplies<>(), twice(), 0);
  test(Iter(ia), Iter(ia + 1), 2, cuda::std::plus<>(), twice(), 4);
  test(Iter(ia), Iter(ia + 2), 0, cuda::std::plus<>(), twice(), 6);
  test(Iter(ia), Iter(ia + 2), 3, cuda::std::multiplies<>(), twice(), 24);
  test(Iter(ia), Iter(ia + sa), 4, cuda::std::multiplies<>(), twice(), 184320); // 64 * 2880
  test(Iter(ia), Iter(ia + sa), 4, cuda::std::plus<>(), twice(), 46);
}

template <typename T, typename Init>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_return_type()
{
  T* p = nullptr;
  unused(p);
  static_assert(
    cuda::std::is_same<Init, decltype(cuda::std::transform_reduce(p, p, Init{}, cuda::std::plus<>(), identity()))>::value,
    "");
}

struct SumMoveOnly
{
  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator()(const MoveOnly& lhs, const MoveOnly& rhs) const noexcept
  {
    return MoveOnly{lhs.get() + rhs.get()};
  }
};

struct TimesTen
{
  __host__ __device__ TEST_CONSTEXPR_CXX14 MoveOnly operator()(const MoveOnly& target) const noexcept
  {
    return MoveOnly{target.get() * 10};
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX14 void test_move_only_types()
{
  MoveOnly ia[] = {{1}, {2}, {3}};
  assert(
    60
    == cuda::std::transform_reduce(cuda::std::begin(ia), cuda::std::end(ia), MoveOnly{0}, SumMoveOnly{}, TimesTen{})
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
  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();
  test<int*>();

  //  Make sure the math is done using the correct type
  {
    auto v       = {1, 2, 3, 4, 5, 6};
    unsigned res = cuda::std::transform_reduce(v.begin(), v.end(), 1U, cuda::std::multiplies<>(), twice());
    assert(res == 46080); // 6! * 64 will not fit into a char
  }

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
