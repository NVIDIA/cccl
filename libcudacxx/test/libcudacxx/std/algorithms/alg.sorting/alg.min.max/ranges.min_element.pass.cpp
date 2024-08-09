//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

//  template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//    indirect_strict_weak_order<projected<I, Proj>> Comp = ranges::less>
//  constexpr I ranges::min_element(I first, S last, Comp comp = {}, Proj proj = {});
//
//  template<forward_range R, class Proj = identity,
//    indirect_strict_weak_order<projected<iterator_t<R>, Proj>> Comp = ranges::less>
//  constexpr borrowed_iterator_t<R> ranges::min_element(R&& r, Comp comp = {}, Proj proj = {});

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <class T>
concept HasMinElement = requires(T t) { cuda::std::ranges::min_element(t); };
#else
template <class T, class = void>
constexpr bool HasMinElement = false;

template <class T>
constexpr bool HasMinElement<T, cuda::std::void_t<decltype(cuda::std::ranges::min_element(cuda::std::declval<T>()))>> =
  true;
#endif

struct NoLessThanOp
{};
struct NotTotallyOrdered
{
  int i;
  __host__ __device__ bool operator<(const NotTotallyOrdered& o) const
  {
    return i < o.i;
  }
};

static_assert(HasMinElement<cuda::std::array<int, 0>>);
static_assert(!HasMinElement<int>);
static_assert(!HasMinElement<NoLessThanOp>);
static_assert(!HasMinElement<NotTotallyOrdered>);

template <class Iter>
__host__ __device__ constexpr void test_iterators(Iter first, Iter last)
{
  decltype(auto) it = cuda::std::ranges::min_element(first, last);
  static_assert(cuda::std::same_as<decltype(it), Iter>);
  if (first != last)
  {
    for (Iter j = first; j != last; ++j)
    {
      assert(!(*j < *it));
    }
  }
  else
  {
    assert(it == first);
  }
}

template <class Range, class Iter>
__host__ __device__ constexpr void test_range(Range&& rng, Iter begin, Iter end)
{
  auto it = cuda::std::ranges::min_element(cuda::std::forward<Range>(rng));
  static_assert(cuda::std::same_as<decltype(it), Iter>);
  if (begin != end)
  {
    for (Iter j = begin; j != end; ++j)
    {
      assert(!(*j < *it));
    }
  }
  else
  {
    assert(it == begin);
  }
}

template <class It>
__host__ __device__ constexpr void test(cuda::std::initializer_list<int> a, int expected)
{
  const int* first = a.begin();
  const int* last  = a.end();
  {
    decltype(auto) it = cuda::std::ranges::min_element(It(first), It(last));
    static_assert(cuda::std::same_as<decltype(it), It>);
    assert(base(it) == first + expected);
  }
  {
    using Sent        = sentinel_wrapper<It>;
    decltype(auto) it = cuda::std::ranges::min_element(It(first), Sent(It(last)));
    static_assert(cuda::std::same_as<decltype(it), It>);
    assert(base(it) == first + expected);
  }
  {
    auto range        = cuda::std::ranges::subrange<It, It>(It(first), It(last));
    decltype(auto) it = cuda::std::ranges::min_element(range);
    static_assert(cuda::std::same_as<decltype(it), It>);
    assert(base(it) == first + expected);
  }
  {
    using Sent        = sentinel_wrapper<It>;
    auto range        = cuda::std::ranges::subrange<It, Sent>(It(first), Sent(It(last)));
    decltype(auto) it = cuda::std::ranges::min_element(range);
    static_assert(cuda::std::same_as<decltype(it), It>);
    assert(base(it) == first + expected);
  }
}

template <class It>
__host__ __device__ constexpr bool test()
{
  test<It>({}, 0);
  test<It>({1}, 0);
  test<It>({1, 2}, 0);
  test<It>({2, 1}, 1);
  test<It>({2, 1, 2}, 1);
  test<It>({2, 1, 1}, 1);

  return true;
}

__host__ __device__ constexpr void test_borrowed_range_and_sentinel()
{
  int a[] = {7, 6, 1, 3, 5, 1, 2, 4};

  int* ret = cuda::std::ranges::min_element(cuda::std::views::all(a));
  assert(ret == a + 2);
  assert(*ret == 1);
}

__host__ __device__ constexpr void test_comparator()
{
  int a[]  = {7, 6, 9, 3, 5, 1, 2, 4};
  int* ret = cuda::std::ranges::min_element(a, cuda::std::ranges::greater{});
  assert(ret == a + 2);
  assert(*ret == 9);
}

__host__ __device__ constexpr void test_projection()
{
  int a[] = {7, 6, 9, 3, 5, 1, 2, 4};
  {
    int* ret = cuda::std::ranges::min_element(a, cuda::std::ranges::less{}, [](int i) {
      return i == 5 ? -100 : i;
    });
    assert(ret == a + 4);
    assert(*ret == 5);
  }
  {
    int* ret = cuda::std::ranges::min_element(a, cuda::std::less<int*>{}, [](int& i) {
      return &i;
    });
    assert(ret == a);
    assert(*ret == 7);
  }
}

struct Immobile
{
  int i;

  __host__ __device__ constexpr Immobile(int i_)
      : i(i_)
  {}
  Immobile(const Immobile&) = delete;
  Immobile(Immobile&&)      = delete;

  __host__ __device__ bool operator==(const Immobile& lhs) const
  {
    return i == lhs.i;
  }
  __host__ __device__ bool operator!=(const Immobile& lhs) const
  {
    return i != lhs.i;
  }

  __host__ __device__ bool operator<(const Immobile& lhs) const
  {
    return i < lhs.i;
  }
  __host__ __device__ bool operator<=(const Immobile& lhs) const
  {
    return i <= lhs.i;
  }
  __host__ __device__ bool operator>(const Immobile& lhs) const
  {
    return i > lhs.i;
  }
  __host__ __device__ bool operator>=(const Immobile& lhs) const
  {
    return i >= lhs.i;
  }
};

__host__ __device__ constexpr void test_immobile()
{
  Immobile arr[] = {1, 2, 3};
  assert(cuda::std::ranges::min_element(arr) == arr);
  assert(cuda::std::ranges::min_element(arr, arr + 3) == arr);
}

__host__ __device__ constexpr void test_dangling()
{
  int compares    = 0;
  int projections = 0;
  auto comparator = [&](int a, int b) {
    ++compares;
    return a < b;
  };
  auto projection = [&](int a) {
    ++projections;
    return a;
  };
  decltype(auto) ret = cuda::std::ranges::min_element(cuda::std::array<int, 3>{1, 2, 3}, comparator, projection);
  static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::dangling>);
  assert(compares == 2);
  assert(projections == 4);
  unused(ret);
}

__host__ __device__ constexpr bool test()
{
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  int a[] = {7, 6, 5, 3, 4, 2, 1, 8};
  test_iterators(a, a + 8);
  int a2[] = {7, 6, 5, 3, 4, 2, 1, 8};
  test_range(a2, a2, a2 + 8);

  test_borrowed_range_and_sentinel();
  test_comparator();
  test_projection();
  test_dangling();

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
