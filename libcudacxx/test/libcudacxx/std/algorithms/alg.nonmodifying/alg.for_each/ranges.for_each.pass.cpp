//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirectly_unary_invocable<projected<I, Proj>> Fun>
//   constexpr ranges::for_each_result<I, Fun>
//     ranges::for_each(I first, S last, Fun f, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirectly_unary_invocable<projected<iterator_t<R>, Proj>> Fun>
//   constexpr ranges::for_each_result<borrowed_iterator_t<R>, Fun>
//     ranges::for_each(R&& r, Fun f, Proj proj = {});

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Callable
{
  __host__ __device__ void operator()(int);
};

template <class Iter, class Sent = Iter, class = void>
inline constexpr bool HasForEachIt = false;
template <class Iter, class Sent>
inline constexpr bool HasForEachIt<Iter,
                                   Sent,
                                   cuda::std::void_t<decltype(cuda::std::ranges::for_each(
                                     cuda::std::declval<Iter>(), cuda::std::declval<Sent>(), Callable{}))>> = true;

static_assert(HasForEachIt<int*>);
static_assert(!HasForEachIt<InputIteratorNotDerivedFrom>);
static_assert(!HasForEachIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasForEachIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasForEachIt<int*, SentinelForNotSemiregular>);
static_assert(!HasForEachIt<int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Func, class = void>
inline constexpr bool HasForEachItFunc = false;
template <class Func>
inline constexpr bool
  HasForEachItFunc<Func,
                   cuda::std::void_t<decltype(cuda::std::ranges::for_each(
                     static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::declval<Func>()))>> = true;

static_assert(HasForEachItFunc<Callable>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotPredicate>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotCopyConstructible>);

template <class Range, class = void>
inline constexpr bool HasForEachR = false;
template <class Range>
inline constexpr bool
  HasForEachR<Range, cuda::std::void_t<decltype(cuda::std::ranges::for_each(cuda::std::declval<Range>(), Callable{}))>> =
    true;

static_assert(HasForEachR<UncheckedRange<int*>>);
static_assert(!HasForEachR<InputRangeNotDerivedFrom>);
static_assert(!HasForEachR<InputRangeNotIndirectlyReadable>);
static_assert(!HasForEachR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasForEachR<InputRangeNotSentinelSemiregular>);
static_assert(!HasForEachR<InputRangeNotSentinelEqualityComparableWith>);

template <class Func, class = void>
inline constexpr bool HasForEachRFunc = false;
template <class Func>
inline constexpr bool HasForEachRFunc<Func,
                                      cuda::std::void_t<decltype(cuda::std::ranges::for_each(
                                        cuda::std::declval<UncheckedRange<int*>>(), cuda::std::declval<Func>()))>> =
  true;

static_assert(HasForEachRFunc<Callable>);
static_assert(!HasForEachRFunc<IndirectUnaryPredicateNotPredicate>);
static_assert(!HasForEachRFunc<IndirectUnaryPredicateNotCopyConstructible>);

struct with_mutable_arg
{
  int i = 0;
  __host__ __device__ constexpr void operator()(int& a) noexcept
  {
    a += i++;
  }
};

struct Always_false
{
  __host__ __device__ constexpr Always_false(const bool val) noexcept
  {
    assert(val);
  }
};

struct should_not_be_called
{
  template <class T>
  __host__ __device__ constexpr void operator()(T&) const noexcept
  {
    Always_false{false};
  }
};

struct zeroes_out
{
  __host__ __device__ constexpr void operator()(int& a) noexcept
  {
    a = 0;
  }
};

template <class Iter, class Sent = Iter>
__host__ __device__ constexpr void test_iterator()
{
  { // simple test
    {
      int a[]            = {1, 6, 3, 4};
      decltype(auto) ret = cuda::std::ranges::for_each(Iter(a), Sent(Iter(a + 4)), with_mutable_arg{});
      static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::for_each_result<Iter, with_mutable_arg>>);
      assert(a[0] == 1);
      assert(a[1] == 7);
      assert(a[2] == 5);
      assert(a[3] == 7);
      assert(base(ret.in) == a + 4);
      int i = 0;
      ret.fun(i);
      assert(i == 4);
    }
    {
      int a[]            = {1, 6, 3, 4};
      auto range         = cuda::std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      decltype(auto) ret = cuda::std::ranges::for_each(range, with_mutable_arg{});
      static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::for_each_result<Iter, with_mutable_arg>>);
      assert(a[0] == 1);
      assert(a[1] == 7);
      assert(a[2] == 5);
      assert(a[3] == 7);
      assert(base(ret.in) == a + 4);
      int i = 0;
      ret.fun(i);
      assert(i == 4);
    }
  }

  { // check that an empty range works
    {
      cuda::std::array<int, 0> a = {};
      cuda::std::ranges::for_each(Iter(a.data()), Sent(Iter(a.data())), should_not_be_called{});
    }
    {
      cuda::std::array<int, 0> a = {};
      auto range                 = cuda::std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data())));
      cuda::std::ranges::for_each(range, should_not_be_called{});
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test_iterator<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterator<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterator<forward_iterator<int*>>();
  test_iterator<bidirectional_iterator<int*>>();
  test_iterator<random_access_iterator<int*>>();
  test_iterator<contiguous_iterator<int*>>();
  test_iterator<int*>();

  { // check that cuda::std::invoke is used
    struct S
    {
      int check;
      int other;
    };
    {
      S a[] = {{1, 2}, {3, 4}, {5, 6}};
      cuda::std::ranges::for_each(a, a + 3, zeroes_out{}, &S::check);
      assert(a[0].check == 0);
      assert(a[0].other == 2);
      assert(a[1].check == 0);
      assert(a[1].other == 4);
      assert(a[2].check == 0);
      assert(a[2].other == 6);
    }
    {
      S a[] = {{1, 2}, {3, 4}, {5, 6}};
      cuda::std::ranges::for_each(a, zeroes_out{}, &S::check);
      assert(a[0].check == 0);
      assert(a[0].other == 2);
      assert(a[1].check == 0);
      assert(a[1].other == 4);
      assert(a[2].check == 0);
      assert(a[2].other == 6);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if !TEST_COMPILER(GCC, <, 10) // accessing value of ‘a’ through a ‘int’ glvalue in a constant expression
  static_assert(test());
#endif // !TEST_COMPILER(GCC, <, 10)

  return 0;
}
