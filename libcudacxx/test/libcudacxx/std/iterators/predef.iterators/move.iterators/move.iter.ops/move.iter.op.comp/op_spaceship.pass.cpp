//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: LIBCUDACXX-has-no-concepts
// XFAIL: c++20

// <iterator>

// move_iterator

// template <class Iter1, three_way_comparable_with<Iter1> Iter2>
//   constexpr auto operator<=>(const move_iterator<Iter1>& x, const move_iterator<Iter2>& y)
//     -> compare_three_way_result_t<Iter1, Iter2>;

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class T, class U>
concept HasEquals = requires(T t, U u) { t == u; };
template <class T, class U>
concept HasSpaceship = requires(T t, U u) { t <=> u; };

static_assert(!HasSpaceship<cuda::std::move_iterator<int*>, cuda::std::move_iterator<char*>>);
static_assert(HasSpaceship<cuda::std::move_iterator<int*>, cuda::std::move_iterator<int*>>);
static_assert(HasSpaceship<cuda::std::move_iterator<int*>, cuda::std::move_iterator<const int*>>);
static_assert(HasSpaceship<cuda::std::move_iterator<const int*>, cuda::std::move_iterator<const int*>>);
static_assert(
  !HasSpaceship<cuda::std::move_iterator<forward_iterator<int*>>, cuda::std::move_iterator<forward_iterator<int*>>>);
static_assert(!HasSpaceship<cuda::std::move_iterator<random_access_iterator<int*>>,
                            cuda::std::move_iterator<random_access_iterator<int*>>>);
static_assert(!HasSpaceship<cuda::std::move_iterator<contiguous_iterator<int*>>,
                            cuda::std::move_iterator<contiguous_iterator<int*>>>);
static_assert(HasSpaceship<cuda::std::move_iterator<three_way_contiguous_iterator<int*>>,
                           cuda::std::move_iterator<three_way_contiguous_iterator<int*>>>);

static_assert(!HasSpaceship<cuda::std::move_iterator<int*>, cuda::std::move_sentinel<int*>>);
static_assert(!HasSpaceship<cuda::std::move_iterator<three_way_contiguous_iterator<int*>>,
                            cuda::std::move_sentinel<three_way_contiguous_iterator<int*>>>);

__host__ __device__ void test_spaceshippable_but_not_three_way_comparable()
{
  struct A
  {
    using value_type      = int;
    using difference_type = int;
    __host__ __device__ int& operator*() const;
    __host__ __device__ A& operator++();
    __host__ __device__ A operator++(int);
    __host__ __device__ cuda::std::strong_ordering operator<=>(const A&) const;
  };
  struct B
  {
    using value_type      = int;
    using difference_type = int;
    __host__ __device__ int& operator*() const;
    __host__ __device__ B& operator++();
    __host__ __device__ B operator++(int);
    __host__ __device__ cuda::std::strong_ordering operator<=>(const B&) const;
    __host__ __device__ bool operator==(const A&) const;
    __host__ __device__ cuda::std::strong_ordering operator<=>(const A&) const;
  };
  static_assert(cuda::std::input_iterator<A>);
  static_assert(cuda::std::input_iterator<B>);
  static_assert(HasEquals<A, B>);
  static_assert(HasSpaceship<A, B>);
  static_assert(!cuda::std::three_way_comparable_with<A, B>);
  static_assert(HasEquals<cuda::std::move_iterator<A>, cuda::std::move_iterator<B>>);
  static_assert(!HasSpaceship<cuda::std::move_iterator<A>, cuda::std::move_iterator<B>>);
}

template <class It, class Jt>
__host__ __device__ constexpr void test_two()
{
  int a[]                               = {3, 1, 4};
  const cuda::std::move_iterator<It> i1 = cuda::std::move_iterator<It>(It(a));
  const cuda::std::move_iterator<It> i2 = cuda::std::move_iterator<It>(It(a + 2));
  const cuda::std::move_iterator<Jt> j1 = cuda::std::move_iterator<Jt>(Jt(a));
  const cuda::std::move_iterator<Jt> j2 = cuda::std::move_iterator<Jt>(Jt(a + 2));
  ASSERT_SAME_TYPE(decltype(i1 <=> i2), cuda::std::strong_ordering);
  assert((i1 <=> i1) == cuda::std::strong_ordering::equal);
  assert((i1 <=> i2) == cuda::std::strong_ordering::less);
  assert((i2 <=> i1) == cuda::std::strong_ordering::greater);
  ASSERT_SAME_TYPE(decltype(i1 <=> j2), cuda::std::strong_ordering);
  assert((i1 <=> j1) == cuda::std::strong_ordering::equal);
  assert((i1 <=> j2) == cuda::std::strong_ordering::less);
  assert((i2 <=> j1) == cuda::std::strong_ordering::greater);
}

__host__ __device__ constexpr bool test()
{
  test_two<int*, int*>();
  test_two<int*, const int*>();
  test_two<const int*, int*>();
  test_two<const int*, const int*>();
  test_two<three_way_contiguous_iterator<int*>, three_way_contiguous_iterator<int*>>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());

  return 0;
}
