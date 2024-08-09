//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<input_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr I ranges::find_if_not(I first, S last, Pred pred, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_iterator_t<R>
//     ranges::find_if_not(R&& r, Pred pred, Proj proj = {});

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

struct Predicate
{
  __host__ __device__ bool operator()(int);
};

#if _TEST_STD_VER > 2017
template <class It, class Sent = It>
concept HasFindIfNotIt = requires(It it, Sent sent) { cuda::std::ranges::find_if_not(it, sent, Predicate{}); };
#else
template <class It, class Sent = It, class = void>
inline constexpr bool HasFindIfNotIt = false;

template <class It, class Sent>
inline constexpr bool
  HasFindIfNotIt<It,
                 Sent,
                 cuda::std::void_t<decltype(cuda::std::ranges::find_if_not(
                   cuda::std::declval<It>(), cuda::std::declval<Sent>(), cuda::std::declval<Predicate>()))>> = true;
#endif

static_assert(HasFindIfNotIt<int*>);
static_assert(!HasFindIfNotIt<InputIteratorNotDerivedFrom>);
static_assert(!HasFindIfNotIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasFindIfNotIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasFindIfNotIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindIfNotIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindIfNotIt<int*, int>);
static_assert(!HasFindIfNotIt<int, int*>);

#if _TEST_STD_VER > 2017
template <class Pred>
concept HasFindIfNotPred = requires(int* it, Pred pred) { cuda::std::ranges::find_if_not(it, it, pred); };
#else
template <class Pred, class = void>
inline constexpr bool HasFindIfNotPred = false;

template <class Pred>
inline constexpr bool
  HasFindIfNotPred<Pred,
                   cuda::std::void_t<decltype(cuda::std::ranges::find_if_not(
                     static_cast<int*>(nullptr), static_cast<int*>(nullptr), cuda::std::declval<Pred>()))>> = true;
#endif

static_assert(!HasFindIfNotPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindIfNotPred<IndirectUnaryPredicateNotPredicate>);

#if _TEST_STD_VER > 2017
template <class R>
concept HasFindIfNotR = requires(R r) { cuda::std::ranges::find_if_not(r, Predicate{}); };
#else
template <class R, class = void>
inline constexpr bool HasFindIfNotR = false;

template <class R>
inline constexpr bool HasFindIfNotR<
  R,
  cuda::std::void_t<decltype(cuda::std::ranges::find_if_not(cuda::std::declval<R>(), cuda::std::declval<Predicate>()))>> =
  true;
#endif

static_assert(HasFindIfNotR<cuda::std::array<int, 0>>);
static_assert(!HasFindIfNotR<int>);
static_assert(!HasFindIfNotR<InputRangeNotDerivedFrom>);
static_assert(!HasFindIfNotR<InputRangeNotIndirectlyReadable>);
static_assert(!HasFindIfNotR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasFindIfNotR<InputRangeNotSentinelSemiregular>);
static_assert(!HasFindIfNotR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
__host__ __device__ constexpr void test_iterators()
{
  {
    int a[]            = {1, 2, 3, 4};
    decltype(auto) ret = cuda::std::ranges::find_if_not(It(a), Sent(It(a + 4)), [c = 0](int) mutable {
      return c++ <= 2;
    });
    static_assert(cuda::std::same_as<decltype(ret), It>);
    assert(base(ret) == a + 3);
    assert(*ret == 4);
  }
  {
    int a[]            = {1, 2, 3, 4};
    auto range         = cuda::std::ranges::subrange<It, Sent>(It(a), Sent(It(a + 4)));
    decltype(auto) ret = cuda::std::ranges::find_if_not(range, [c = 0](int) mutable {
      return c++ <= 2;
    });
    static_assert(cuda::std::same_as<decltype(ret), It>);
    assert(base(ret) == a + 3);
    assert(*ret == 4);
  }
}

struct NonConstComparableLValue
{
  __host__ __device__ friend constexpr bool operator==(const NonConstComparableLValue&, const NonConstComparableLValue&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator==(NonConstComparableLValue&, NonConstComparableLValue&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator==(const NonConstComparableLValue&, NonConstComparableLValue&)
  {
    return false;
  }
  __host__ __device__ friend constexpr bool operator==(NonConstComparableLValue&, const NonConstComparableLValue&)
  {
    return true;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(const NonConstComparableLValue&, const NonConstComparableLValue&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(NonConstComparableLValue&, NonConstComparableLValue&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(const NonConstComparableLValue&, NonConstComparableLValue&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(NonConstComparableLValue&, const NonConstComparableLValue&)
  {
    return false;
  }
#endif
};

struct AlwaysFalse
{
  __host__ __device__ constexpr bool operator()(int) const
  {
    return false;
  }
};
struct AlwaysTrue
{
  __host__ __device__ constexpr bool operator()(int) const
  {
    return true;
  }
};
struct CheckStar
{
  template <class T>
  __host__ __device__ constexpr bool operator()(T&& e) const
  {
    return e != NonConstComparableLValue{};
  }
};

__host__ __device__ constexpr bool test()
{
  test_iterators<int*>();
  test_iterators<const int*>();
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  { // check that projections are used properly and that they are called with the iterator directly
    struct ToAddress
    {
      __host__ __device__ constexpr int* operator()(int& i) const
      {
        return &i;
      }
    };
    struct PointsToLast
    {
      int* a;
      __host__ __device__ constexpr bool operator()(int* i) const
      {
        return i != a + 3;
      }
    };
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_if_not(a, a + 4, PointsToLast{a}, ToAddress{});
      assert(ret == a + 3);
    }
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_if_not(a, PointsToLast{a}, ToAddress{});
      assert(ret == a + 3);
    }
  }

  {
    // check that the first element is returned
    struct NotZero
    {
      __host__ __device__ constexpr bool operator()(int i) const
      {
        return i != 0;
      }
    };
    {
      struct S
      {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = cuda::std::ranges::find_if_not(a, NotZero{}, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
    {
      struct S
      {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = cuda::std::ranges::find_if_not(a, a + 3, NotZero{}, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
  }

  {
    // check that end + 1 iterator is returned with no match
    {
      int a[]  = {1, 1, 1};
      auto ret = cuda::std::ranges::find_if(a, a + 3, AlwaysFalse{});
      assert(ret == a + 3);
    }
    {
      int a[]  = {1, 1, 1};
      auto ret = cuda::std::ranges::find_if(a, AlwaysFalse{});
      assert(ret == a + 3);
    }
  }

  { // check that ranges::dangling is returned
    decltype(auto) ret = cuda::std::ranges::find_if_not(cuda::std::array<int, 2>{1, 2}, AlwaysTrue{});
    static_assert(cuda::std::same_as<decltype(ret), cuda::std::ranges::dangling>);
    unused(ret);
  }

  { // check that an iterator is returned with a borrowing range
    int a[]            = {1, 2, 3, 4};
    decltype(auto) ret = cuda::std::ranges::find_if_not(cuda::std::views::all(a), AlwaysFalse{});
    static_assert(cuda::std::same_as<decltype(ret), int*>);
    assert(ret == a);
    assert(*ret == 1);
  }

  { // check that cuda::std::invoke is used
    struct S
    {
      int i;
    };
    S a[]              = {S{1}, S{3}, S{2}};
    decltype(auto) ret = cuda::std::ranges::find_if_not(a, AlwaysTrue{}, &S::i);
    static_assert(cuda::std::same_as<decltype(ret), S*>);
    assert(ret == a + 3);
  }

  { // count projection and predicate invocation count
    struct CountPredicate
    {
      int& predicate_count;
      __host__ __device__ constexpr bool operator()(int i) const
      {
        ++predicate_count;
        return i != 2;
      }
    };
    struct CountProjection
    {
      int& projection_count;
      __host__ __device__ constexpr int operator()(int i) const
      {
        ++projection_count;
        return i;
      }
    };

    {
      int a[]              = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;
      auto ret =
        cuda::std::ranges::find_if_not(a, a + 4, CountPredicate{predicate_count}, CountProjection{projection_count});
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(predicate_count == 2);
      assert(projection_count == 2);
    }
    {
      int a[]              = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;
      auto ret = cuda::std::ranges::find_if_not(a, CountPredicate{predicate_count}, CountProjection{projection_count});
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(predicate_count == 2);
      assert(projection_count == 2);
    }
  }

  { // check that the return type of `iter::operator*` doesn't change
    {
      NonConstComparableLValue a[] = {NonConstComparableLValue{}};
      auto ret                     = cuda::std::ranges::find_if_not(a, a + 1, CheckStar{});
      assert(ret == a);
    }
    {
      NonConstComparableLValue a[] = {NonConstComparableLValue{}};
      auto ret                     = cuda::std::ranges::find_if_not(a, CheckStar{});
      assert(ret == a);
    }
  }

#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  if (!__builtin_is_constant_evaluated())
  {
    // check that an empty range works
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_if_not(a.begin(), a.end(), AlwaysTrue{});
      assert(ret == a.begin());
    }
    {
      cuda::std::array<int, 0> a = {};
      auto ret                   = cuda::std::ranges::find_if_not(a, AlwaysTrue{});
      assert(ret == a.begin());
    }
  }
#endif

  {
    // check that the implicit conversion to bool works
    struct ReturnBooleanTestable
    {
      __host__ __device__ constexpr BooleanTestable operator()(const int& i) const
      {
        return BooleanTestable{i != 3};
      }
    };
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_if_not(a, a + 4, ReturnBooleanTestable{});
      assert(ret == a + 2);
    }
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = cuda::std::ranges::find_if_not(a, ReturnBooleanTestable{});
      assert(ret == a + 2);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
