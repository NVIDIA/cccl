//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>

// Became constexpr in C++20
// template<class InputIterator, class OutputIterator, class T>
//     OutputIterator exclusive_scan(InputIterator first, InputIterator last,
//                                   OutputIterator result, T init);
//

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/numeric>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter1, class T>
__host__ __device__ constexpr void test(Iter1 first, Iter1 last, T init, const T* rFirst, const T* rLast)
{
  assert((rLast - rFirst) <= 5); // or else increase the size of "out"
  T out[5] = {};

  // Not in place
  T* end = cuda::std::exclusive_scan(first, last, out, init);
  assert(cuda::std::equal(out, end, rFirst, rLast));

  // In place
  // manual copy to pacify constexpr evaluators
  for (auto out2 = out; first != last; ++first, (void) ++out2)
  {
    *out2 = *first;
  }
  end = cuda::std::exclusive_scan(out, end, out, init);
  assert(cuda::std::equal(out, end, rFirst, rLast));
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  int ia[]          = {1, 3, 5, 7, 9};
  const int pRes[]  = {0, 1, 4, 9, 16};
  const unsigned sa = sizeof(ia) / sizeof(ia[0]);
  static_assert(sa == sizeof(pRes) / sizeof(pRes[0]), ""); // just to be sure

  for (unsigned int i = 0; i < sa; ++i)
  {
    test(Iter(ia), Iter(ia + i), 0, pRes, pRes + i);
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
    cuda::std::exclusive_scan(v.begin(), v.end(), v.begin(), cuda::std::size_t{50});
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      assert(v[i] == 50 + i * 3);
    }
  }

  {
    cuda::std::array<cuda::std::size_t, 10> v{};
    cuda::std::iota(v.begin(), v.end(), 0);
    cuda::std::exclusive_scan(v.begin(), v.end(), v.begin(), cuda::std::size_t{30});
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      assert(v[i] == 30 + triangle(i - 1));
    }
  }

  {
    cuda::std::array<cuda::std::size_t, 10> v{};
    cuda::std::iota(v.begin(), v.end(), 1);
    cuda::std::exclusive_scan(v.begin(), v.end(), v.begin(), cuda::std::size_t{40});
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      assert(v[i] == 40 + triangle(i));
    }
  }
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
