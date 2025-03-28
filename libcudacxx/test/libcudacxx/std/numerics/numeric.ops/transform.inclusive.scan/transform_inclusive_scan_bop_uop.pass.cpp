//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template<class InputIterator, class OutputIterator, class T,
//          class BinaryOperation, class UnaryOperation>
//   OutputIterator transform_inclusive_scan(InputIterator first, InputIterator last,
//                                           OutputIterator result,
//                                           BinaryOperation binary_op,
//                                           UnaryOperation unary_op);

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

struct add_one
{
  template <typename T>
  __host__ __device__ constexpr T operator()(T x) const
  {
    return x + 1;
  }
};

template <class Iter1, class BOp, class UOp, class T>
__host__ __device__ constexpr void test(Iter1 first, Iter1 last, BOp bop, UOp uop, const T* rFirst, const T* rLast)
{
  assert((rLast - rFirst) <= 5); // or else increase the size of "out"
  T out[5] = {};

  // Not in place
  T* end = cuda::std::transform_inclusive_scan(first, last, out, bop, uop);
  assert(cuda::std::equal(out, end, rFirst, rLast));

  // In place
  // manual copy to pacify constexpr evaluators
  for (auto out2 = out; first != last; ++first, (void) ++out2)
  {
    *out2 = *first;
  }
  end = cuda::std::transform_inclusive_scan(out, end, out, bop, uop);
  assert(cuda::std::equal(out, end, rFirst, rLast));
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]           = {1, 3, 5, 7, 9};
  const int pResI0[] = {2, 6, 12, 20, 30}; // with add_one
  const int mResI0[] = {2, 8, 48, 384, 3840};
  const int pResN0[] = {-1, -4, -9, -16, -25}; // with negate
  const int mResN0[] = {-1, 3, -15, 105, -945};
  const unsigned sa  = sizeof(ia) / sizeof(ia[0]);
  static_assert(sa == sizeof(pResI0) / sizeof(pResI0[0]), ""); // just to be sure
  static_assert(sa == sizeof(mResI0) / sizeof(mResI0[0]), ""); // just to be sure
  static_assert(sa == sizeof(pResN0) / sizeof(pResN0[0]), ""); // just to be sure
  static_assert(sa == sizeof(mResN0) / sizeof(mResN0[0]), ""); // just to be sure

  for (unsigned int i = 0; i < sa; ++i)
  {
    test(Iter(ia), Iter(ia + i), cuda::std::plus<>(), add_one{}, pResI0, pResI0 + i);
    test(Iter(ia), Iter(ia + i), cuda::std::multiplies<>(), add_one{}, mResI0, mResI0 + i);
    test(Iter(ia), Iter(ia + i), cuda::std::plus<>(), cuda::std::negate<>(), pResN0, pResN0 + i);
    test(Iter(ia), Iter(ia + i), cuda::std::multiplies<>(), cuda::std::negate<>(), mResN0, mResN0 + i);
  }
}

__host__ __device__ constexpr cuda::std::size_t triangle(size_t n)
{
  return n * (n + 1) / 2;
}

//  Basic sanity
__host__ __device__ constexpr void basic_tests()
{
  {
    cuda::std::array<cuda::std::size_t, 10> v{};
    cuda::std::fill(v.begin(), v.end(), 3);
    cuda::std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), cuda::std::plus<>(), add_one{});
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      assert(v[i] == (i + 1) * 4);
    }
  }

  {
    cuda::std::array<cuda::std::size_t, 10> v{};
    cuda::std::iota(v.begin(), v.end(), 0);
    cuda::std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), cuda::std::plus<>(), add_one{});
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      assert(v[i] == triangle(i) + i + 1);
    }
  }

  {
    cuda::std::array<cuda::std::size_t, 10> v{};
    cuda::std::iota(v.begin(), v.end(), 1);
    cuda::std::transform_inclusive_scan(v.begin(), v.end(), v.begin(), cuda::std::plus<>(), add_one{});
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      assert(v[i] == triangle(i + 1) + i + 1);
    }
  }

#if !TEST_COMPILER(NVHPC) // NVHPC seems unable to silence the warning
  TEST_NV_DIAG_SUPPRESS(expr_has_no_effect)
  {
    cuda::std::array<cuda::std::size_t, 0> v{};
    cuda::std::array<cuda::std::size_t, 0> res{};
    cuda::std::transform_inclusive_scan(v.begin(), v.end(), res.begin(), cuda::std::plus<>(), add_one{});
    assert(res.empty());
  }
#endif // !TEST_COMPILER(NVHPC)
}

__host__ __device__ constexpr bool test()
{
  basic_tests();

  //  All the iterator categories
  test<cpp17_input_iterator<const int*>>();
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
