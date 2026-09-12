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
//   constexpr subrange<I> ranges::find_last_if_not(I first, S last, Pred pred, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_subrange_t<R> ranges::find_last_if_not(R&& r, Pred pred, Proj proj = {});

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
concept HasFindLastIfNotIt = requires(It it, Sent sent) { cuda::std::ranges::find_last_if_not(it, sent, Predicate{}); };
#else
template <class It, class Sent = It, class = void>
inline constexpr bool HasFindLastIfNotIt = false;

template <class It, class Sent>
inline constexpr bool
  HasFindLastIfNotIt<It,
                     Sent,
                     cuda::std::void_t<decltype(cuda::std::ranges::find_last_if_not(
                       cuda::std::declval<It>(), cuda::std::declval<Sent>(), cuda::std::declval<Predicate>()))>> = true;
#endif

static_assert(HasFindLastIfNotIt<int*>);
static_assert(HasFindLastIfNotIt<forward_iterator<int*>>);
// find_last_if_not requires a forward iterator, an input iterator is not enough
static_assert(!HasFindLastIfNotIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);
static_assert(!HasFindLastIfNotIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIfNotIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIfNotIt<forward_iterator<int*>, SentinelForNotSemiregular>);

static_assert(!HasFindLastIfNotIt<int*, int>);
static_assert(!HasFindLastIfNotIt<int, int*>);

#if TEST_STD_VER > 2017
template <class Pred>
concept HasFindLastIfNotPred = requires(int* it, Pred pred) { cuda::std::ranges::find_last_if_not(it, it, pred); };
#else
template <class Pred, class = void>
inline constexpr bool HasFindLastIfNotPred = false;

template <class Pred>
inline constexpr bool
  HasFindLastIfNotPred<Pred,
                       cuda::std::void_t<decltype(cuda::std::ranges::find_last_if_not(
                         static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::declval<Pred>()))>> = true;
#endif

static_assert(!HasFindLastIfNotPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindLastIfNotPred<IndirectUnaryPredicateNotPredicate>);

#if TEST_STD_VER > 2017
template <class R>
concept HasFindLastIfNotR = requires(R r) { cuda::std::ranges::find_last_if_not(r, Predicate{}); };
#else
template <class R, class = void>
inline constexpr bool HasFindLastIfNotR = false;

template <class R>
inline constexpr bool HasFindLastIfNotR<R,
                                        cuda::std::void_t<decltype(cuda::std::ranges::find_last_if_not(
                                          cuda::std::declval<R>(), cuda::std::declval<Predicate>()))>> = true;
#endif

static_assert(HasFindLastIfNotR<cuda::std::array<int, 0>>);
static_assert(!HasFindLastIfNotR<int>);
static_assert(!HasFindLastIfNotR<ForwardRangeNotDerivedFrom>);
static_assert(!HasFindLastIfNotR<ForwardRangeNotIncrementable>);
static_assert(!HasFindLastIfNotR<ForwardRangeNotSentinelSemiregular>);

struct IsNotTwo
{
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int i) const
  {
    return i != 2;
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
    // the *last* element not satisfying the predicate is reported
    int a[]            = {1, 2, 3, 2, 5};
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(It(a), Sent(It(a + 5)), IsNotTwo{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<It>>);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 5);
    assert(*ret.begin() == 2);
  }
  {
    int a[]            = {1, 2, 3, 2, 5};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a + 5)));
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(range, IsNotTwo{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<It>>);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 5);
  }
  {
    // no match yields an empty subrange at the end of the range
    int a[]            = {1, 1, 1};
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(It(a), Sent(It(a + 3)), AlwaysTrue{});
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 3);
    assert(ret.empty());
  }
  {
    // an empty range yields an empty subrange, and the sentinel is honoured rather than the underlying storage
    int a[]            = {2, 2, 2};
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(It(a), Sent(It(a)), IsNotTwo{});
    assert(base(ret.begin()) == a);
    assert(base(ret.end()) == a);
    assert(ret.empty());
  }
  {
    int a[]            = {2, 2, 2};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a)));
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(range, IsNotTwo{});
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

struct IsNotZero
{
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(int i) const
  {
    return i != 0;
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
    struct DoesNotPointToFirst
    {
      int* a;
      TEST_HOST_DEVICE_FUNC constexpr bool operator()(int* i) const
      {
        return i != a;
      }
    };
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_last_if_not(a, a + 4, DoesNotPointToFirst{a}, ToAddress{});
      assert(ret.begin() == a);
      assert(ret.end() == a + 4);
    }
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_last_if_not(a, DoesNotPointToFirst{a}, ToAddress{});
      assert(ret.begin() == a);
      assert(ret.end() == a + 4);
    }
  }

  {
    // check that the last of several matches is returned, not the first
    S a[]              = {{0, 1}, {0, 2}, {0, 3}};
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(a, IsNotZero{}, &S::comp);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<S*>>);
    assert(ret.begin() == a + 2);
    assert(ret.end() == a + 3);
    assert(ret.begin()->other == 3);
  }

  {
    // check that ranges::dangling is returned
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(cuda::std::array<int, 2>{1, 2}, AlwaysFalse{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::dangling>);
    unused(ret);
  }

  {
    // check that a subrange is returned for a borrowing range
    int a[]            = {1, 2, 3, 4};
    decltype(auto) ret = cuda::std::ranges::find_last_if_not(cuda::std::views::all(a), AlwaysFalse{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<int*>>);
    assert(ret.begin() == a + 3);
    assert(ret.end() == a + 4);
  }

  if (!TEST_IS_CONSTANT_EVALUATED())
  {
    // check that an empty range works
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_last_if_not(a.begin(), a.end(), AlwaysFalse{});
      assert(ret.begin() == a.begin());
      assert(ret.end() == a.begin());
    }
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_last_if_not(a, AlwaysFalse{});
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
        return BooleanTestable{i != 3};
      }
    };
    {
      int a[]  = {1, 3, 2, 3};
      auto ret = cuda::std::ranges::find_last_if_not(a, a + 4, ReturnBooleanTestable{});
      assert(ret.begin() == a + 3);
    }
    {
      int a[]  = {1, 3, 2, 3};
      auto ret = cuda::std::ranges::find_last_if_not(a, ReturnBooleanTestable{});
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
