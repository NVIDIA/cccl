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

// template<forward_iterator I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr subrange<I> ranges::find_last(I first, S last, const T& value, Proj proj = {});
// template<forward_range R, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//   constexpr borrowed_subrange_t<R> ranges::find_last(R&& r, const T& value, Proj proj = {});

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <class It, class Sent = It>
concept HasFindLastIt = requires(It it, Sent sent) { cuda::std::ranges::find_last(it, sent, *it); };
#else
template <class It, class Sent = It, class = void>
inline constexpr bool HasFindLastIt = false;

template <class It, class Sent>
inline constexpr bool
  HasFindLastIt<It,
                Sent,
                cuda::std::void_t<decltype(cuda::std::ranges::find_last(
                  cuda::std::declval<It>(), cuda::std::declval<Sent>(), *cuda::std::declval<It>()))>> = true;
#endif

static_assert(HasFindLastIt<int*>);
static_assert(HasFindLastIt<forward_iterator<int*>>);
// find_last requires a forward iterator, an input iterator is not enough
static_assert(!HasFindLastIt<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>);
static_assert(!HasFindLastIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIt<forward_iterator<int*>, SentinelForNotSemiregular>);

static_assert(!HasFindLastIt<int*, int>);
static_assert(!HasFindLastIt<int, int*>);

#if TEST_STD_VER > 2017
template <class R>
concept HasFindLastR = requires(R r) { cuda::std::ranges::find_last(r, 0); };
#else
template <class R, class = void>
inline constexpr bool HasFindLastR = false;

template <class R>
inline constexpr bool
  HasFindLastR<R, cuda::std::void_t<decltype(cuda::std::ranges::find_last(cuda::std::declval<R>(), 0))>> = true;
#endif

static_assert(HasFindLastR<cuda::std::array<int, 0>>);
static_assert(!HasFindLastR<int>);
static_assert(!HasFindLastR<ForwardRangeNotDerivedFrom>);
static_assert(!HasFindLastR<ForwardRangeNotIncrementable>);
static_assert(!HasFindLastR<ForwardRangeNotSentinelSemiregular>);

template <class It, class Sent = It>
TEST_HOST_DEVICE_FUNC constexpr void test_iterators()
{
  {
    // the *last* match is reported, together with the end of the range
    int a[]            = {1, 2, 3, 2, 5};
    decltype(auto) ret = cuda::std::ranges::find_last(It(a), Sent(It(a + 5)), 2);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<It>>);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 5);
    assert(*ret.begin() == 2);
  }
  {
    int a[]            = {1, 2, 3, 2, 5};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a + 5)));
    decltype(auto) ret = cuda::std::ranges::find_last(range, 2);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<It>>);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 5);
  }
  {
    // no match yields an empty subrange at the end of the range
    int a[]            = {1, 2, 3};
    decltype(auto) ret = cuda::std::ranges::find_last(It(a), Sent(It(a + 3)), 4);
    assert(base(ret.begin()) == a + 3);
    assert(base(ret.end()) == a + 3);
    assert(ret.empty());
  }
  {
    // an empty range yields an empty subrange, and the sentinel is honoured rather than the underlying storage
    int a[]            = {1, 2, 3};
    decltype(auto) ret = cuda::std::ranges::find_last(It(a), Sent(It(a)), 1);
    assert(base(ret.begin()) == a);
    assert(base(ret.end()) == a);
    assert(ret.empty());
  }
  {
    int a[]            = {1, 2, 3};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a)));
    decltype(auto) ret = cuda::std::ranges::find_last(range, 1);
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

struct Widget
{
  int value;

  TEST_HOST_DEVICE_FUNC friend constexpr bool operator==(const Widget& lhs, const Widget& rhs)
  {
    return lhs.value == rhs.value;
  }
#if TEST_STD_VER < 2020
  TEST_HOST_DEVICE_FUNC friend constexpr bool operator!=(const Widget& lhs, const Widget& rhs)
  {
    return lhs.value != rhs.value;
  }
#endif // TEST_STD_VER < 2020
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
    // check that the last of several matches is returned, not the first
    S a[]              = {{0, 1}, {0, 2}, {0, 3}};
    decltype(auto) ret = cuda::std::ranges::find_last(a, 0, &S::comp);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<S*>>);
    assert(ret.begin() == a + 2);
    assert(ret.end() == a + 3);
    assert(ret.begin()->other == 3);
  }
  {
    // the same through the iterator overload
    S a[]    = {{0, 1}, {0, 2}, {0, 3}};
    auto ret = cuda::std::ranges::find_last(a, a + 3, 0, &S::comp);
    assert(ret.begin() == a + 2);
    assert(ret.end() == a + 3);
  }

  {
    // check that ranges::dangling is returned
    decltype(auto) ret = cuda::std::ranges::find_last(cuda::std::array<int, 2>{1, 2}, 1);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::dangling>);
    unused(ret);
  }

  {
    // check that a subrange is returned for a borrowing range
    int a[]            = {1, 2, 3, 2};
    decltype(auto) ret = cuda::std::ranges::find_last(cuda::std::views::all(a), 2);
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::subrange<int*>>);
    assert(ret.begin() == a + 3);
    assert(ret.end() == a + 4);
  }

  {
    // check that a class type with a user defined operator== works
    Widget a[] = {Widget{1}, Widget{2}, Widget{1}};
    auto ret   = cuda::std::ranges::find_last(a, Widget{1});
    assert(ret.begin() == a + 2);
    assert(ret.end() == a + 3);
  }

  if (!TEST_IS_CONSTANT_EVALUATED())
  {
    // check that an empty range works
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_last(a.begin(), a.end(), 1);
      assert(ret.begin() == a.begin());
      assert(ret.end() == a.begin());
    }
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_last(a, 1);
      assert(ret.begin() == a.begin());
      assert(ret.end() == a.begin());
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
