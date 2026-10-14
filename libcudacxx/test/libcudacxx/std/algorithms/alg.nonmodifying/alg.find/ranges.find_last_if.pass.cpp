//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: a non-__tile__ variable cannot be used in tile code

// template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr subrange<I> ranges::find_last_if(I first, S last, Pred pred, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_subrange_t<R> ranges::find_last_if(R&& r, Pred pred, Proj proj = {});

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"
#include "test_macros.h"

struct Predicate
{
  TEST_HOST_DEVICE_FUNC bool operator()(int);
};

#if TEST_STD_VER > 2017
template <class It, class Sent = It>
concept HasFindLastIfIt = requires(It it, Sent sent) { cuda::std::ranges::find_last_if(it, sent, Predicate{}); };
#else
template <class It, class Sent = It, class = void>
inline constexpr bool HasFindLastIfIt = false;

template <class It, class Sent>
inline constexpr bool
  HasFindLastIfIt<It,
                  Sent,
                  cuda::std::void_t<decltype(cuda::std::ranges::find_last_if(
                    cuda::std::declval<It>(), cuda::std::declval<Sent>(), cuda::std::declval<Predicate>()))>> = true;
#endif

static_assert(HasFindLastIfIt<int*>);
static_assert(HasFindLastIfIt<forward_iterator<int*>>);
// find_last_if requires a forward iterator, an input iterator is not enough
static_assert(!HasFindLastIfIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);
static_assert(!HasFindLastIfIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIfIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIfIt<forward_iterator<int*>, SentinelForNotSemiregular>);

static_assert(!HasFindLastIfIt<int*, int>);
static_assert(!HasFindLastIfIt<int, int*>);

#if TEST_STD_VER > 2017
template <class Pred>
concept HasFindLastIfPred = requires(int* it, Pred pred) { cuda::std::ranges::find_last_if(it, it, pred); };
#else
template <class Pred, class = void>
inline constexpr bool HasFindLastIfPred = false;

template <class Pred>
inline constexpr bool
  HasFindLastIfPred<Pred,
                    cuda::std::void_t<decltype(cuda::std::ranges::find_last_if(
                      static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::declval<Pred>()))>> = true;
#endif

static_assert(!HasFindLastIfPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindLastIfPred<IndirectUnaryPredicateNotPredicate>);

#if TEST_STD_VER > 2017
template <class R>
concept HasFindLastIfR = requires(R r) { cuda::std::ranges::find_last_if(r, Predicate{}); };
#else
template <class R, class = void>
inline constexpr bool HasFindLastIfR = false;

template <class R>
inline constexpr bool HasFindLastIfR<R,
                                     cuda::std::void_t<decltype(cuda::std::ranges::find_last_if(
                                       cuda::std::declval<R>(), cuda::std::declval<Predicate>()))>> = true;
#endif

static_assert(HasFindLastIfR<cuda::std::array<int, 0>>);
static_assert(!HasFindLastIfR<int>);
static_assert(!HasFindLastIfR<ForwardRangeNotDerivedFrom>);
static_assert(!HasFindLastIfR<ForwardRangeNotIncrementable>);
static_assert(!HasFindLastIfR<ForwardRangeNotSentinelSemiregular>);

struct IsTwo
{
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int i) const
  {
    return i == 2;
  }
};
struct AlwaysFalse
{
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int) const
  {
    return false;
  }
};
struct AlwaysTrue
{
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int) const
  {
    return true;
  }
};

template <class It, class Sent = It>
TEST_HOST_DEVICE_FUNC constexpr void test_iterators()
{
  {
    // the *last* match is reported, together with the end of the range
    int a[]            = {1, 2, 3, 2, 5};
    decltype(auto) ret = cuda::std::ranges::find_last_if(It(a), Sent(It(a + 5)), IsTwo{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<It>>);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 5);
    assert(*ret.begin() == 2);
  }
  {
    int a[]            = {1, 2, 3, 2, 5};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a + 5)));
    decltype(auto) ret = cuda::std::ranges::find_last_if(range, IsTwo{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<It>>);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 5);
  }
  {
    // no match yields an empty subrange at the end of the range
    int a[]            = {1, 1, 1};
    decltype(auto) ret = cuda::std::ranges::find_last_if(It(a), Sent(It(a + 3)), AlwaysFalse{});
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 3);
    assert(ret.empty());
  }
  {
    // an empty range yields an empty subrange, and the sentinel is honoured rather than the underlying storage
    int a[]            = {2, 2, 2};
    decltype(auto) ret = cuda::std::ranges::find_last_if(It(a), Sent(It(a)), IsTwo{});
    assert(base(ret.begin()) == a);
    assert(base(ret.end()) == a);
    assert(ret.empty());
  }
  {
    int a[]            = {2, 2, 2};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a)));
    decltype(auto) ret = cuda::std::ranges::find_last_if(range, IsTwo{});
    assert(base(ret.begin()) == a);
    assert(base(ret.end()) == a);
    assert(ret.empty());
  }
}

struct S
{
  int comp;
  int other;
};

struct IsZero
{
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int i) const
  {
    return i == 0;
  }
};

struct CountPredicate
{
  int& predicate_count;
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int i) const
  {
    ++predicate_count;
    return i == 2;
  }
};

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_iterators<int*>();
  test_iterators<const int*>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>();

  {
    // check that projections are used properly and that they are called with the iterator directly
    struct ToAddress
    {
      TEST_HOST_DEVICE_FUNC constexpr int* operator()(int& i) const
      {
        return &i;
      }
    };
    struct PointsToFirst
    {
      int* a;
      TEST_HOST_DEVICE_FUNC constexpr bool operator()(int* i) const
      {
        return i == a;
      }
    };
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_last_if(a, a + 4, PointsToFirst{a}, ToAddress{});
      assert(ret.begin() == a);
      assert(ret.end() == a + 4);
    }
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_last_if(a, PointsToFirst{a}, ToAddress{});
      assert(ret.begin() == a);
      assert(ret.end() == a + 4);
    }
  }

  {
    // check that the last of several matches is returned, not the first
    S a[]              = {{0, 1}, {0, 2}, {0, 3}};
    decltype(auto) ret = cuda::std::ranges::find_last_if(a, IsZero{}, &S::comp);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<S*>>);
    assert(ret.begin() == a + 2);
    assert(ret.end() == a + 3);
    assert(ret.begin()->other == 3);
  }

  {
    // check that ranges::dangling is returned
    decltype(auto) ret = cuda::std::ranges::find_last_if(cuda::std::array<int, 2>{1, 2}, AlwaysTrue{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::dangling>);
    unused(ret);
  }

  {
    // check that a subrange is returned for a borrowing range
    int a[]            = {1, 2, 3, 4};
    decltype(auto) ret = cuda::std::ranges::find_last_if(cuda::std::views::all(a), AlwaysTrue{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<int*>>);
    assert(ret.begin() == a + 3);
    assert(ret.end() == a + 4);
  }

  {
    // a bidirectional iterator only walks back from the end up to the match
    int a[]             = {2, 1, 2, 1};
    int predicate_count = 0;
    auto ret            = cuda::std::ranges::find_last_if(a, a + 4, CountPredicate{predicate_count});
    assert(ret.begin() == a + 2);
    assert(predicate_count == 2);
  }
  {
    // a forward iterator has to traverse the whole range
    int a[]             = {2, 1, 2, 1};
    int predicate_count = 0;
    auto ret            = cuda::std::ranges::find_last_if(
      forward_iterator<int*>(a), forward_iterator<int*>(a + 4), CountPredicate{predicate_count});
    assert(base(ret.begin()) == a + 2);
    assert(predicate_count == 4);
  }

  if (!TEST_IS_CONSTANT_EVALUATED())
  {
    // check that an empty range works
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_last_if(a.begin(), a.end(), AlwaysTrue{});
      assert(ret.begin() == a.begin());
      assert(ret.end() == a.begin());
    }
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_last_if(a, AlwaysTrue{});
      assert(ret.begin() == a.begin());
      assert(ret.end() == a.begin());
    }
  }

  {
    // check that the implicit conversion to bool works
    struct ReturnBooleanTestable
    {
      TEST_HOST_DEVICE_FUNC constexpr BooleanTestable operator()(const int& i) const
      {
        return BooleanTestable{i == 3};
      }
    };
    {
      int a[]  = {1, 3, 2, 3};
      auto ret = cuda::std::ranges::find_last_if(a, a + 4, ReturnBooleanTestable{});
      assert(ret.begin() == a + 3);
    }
    {
      int a[]  = {1, 3, 2, 3};
      auto ret = cuda::std::ranges::find_last_if(a, ReturnBooleanTestable{});
      assert(ret.begin() == a + 3);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
