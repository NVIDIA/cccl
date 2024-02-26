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

// constexpr auto begin();

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const
  {
    return nullptr;
  }
  __host__ __device__ int* end() const
  {
    return nullptr;
  }
};

// Test that begin is not const
#if TEST_STD_VER >= 2020
template <class T>
concept HasBegin = requires(T t) { t.begin(); };
#else
template <class T, class = void>
inline constexpr bool HasBegin = false;

template <class T>
inline constexpr bool HasBegin<T, cuda::std::void_t<decltype(cuda::std::declval<T>().begin())>> = true;
#endif

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 3;
  }
};

static_assert(HasBegin<cuda::std::ranges::drop_while_view<View, Pred>>);
static_assert(!HasBegin<const cuda::std::ranges::drop_while_view<View, Pred>>);

template <bool Ret>
struct always
{
  template <class... Args>
  __host__ __device__ constexpr bool operator()(Args&&...) const
  {
    return Ret;
  }
};

struct TrackingPred
{
  __host__ __device__ constexpr explicit TrackingPred(bool* moved, bool* copied)
      : moved_(moved)
      , copied_(copied)
  {}
  __host__ __device__ constexpr TrackingPred(TrackingPred const& other)
      : moved_(other.moved_)
      , copied_(other.copied_)
  {
    *copied_ = true;
  }
  __host__ __device__ constexpr TrackingPred(TrackingPred&& other)
      : moved_(other.moved_)
      , copied_(other.copied_)
  {
    *moved_ = true;
  }
  TrackingPred& operator=(TrackingPred const&) = default;
  TrackingPred& operator=(TrackingPred&&)      = default;

  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 3;
  }
  bool* moved_;
  bool* copied_;
};

template <class Range, class Iter, class Sent, cuda::std::size_t N>
__host__ __device__ constexpr auto make_subrange(int (&buffer)[N])
{
  return Range{Iter{buffer}, Sent{Iter{buffer + N}}};
}

template <class Iter>
__host__ __device__ constexpr void testOne()
{
  using Sent  = sentinel_wrapper<Iter>;
  using Range = cuda::std::ranges::subrange<Iter, Sent>;

  // empty
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  if (!__builtin_is_constant_evaluated())
  {
    cuda::std::array<int, 0> a;
    Range range{Iter{a.data()}, Sent{Iter{a.data() + a.size()}}};
    cuda::std::ranges::drop_while_view<Range, always<false>> dwv{cuda::std::move(range), always<false>{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == a.data() + a.size());
  }
#endif

  // 1 element not dropped
  {
    int buffer[] = {1};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), always<false>> dwv{cuda::std::move(range), always<false>{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == buffer);
  }

  // 1 element dropped
  {
    int buffer[] = {1};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), always<true>> dwv{cuda::std::move(range), always<true>{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == buffer + 1);
  }

  // multiple elements. no element dropped
  {
    int buffer[] = {1, 2, 3, 4, 5};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), always<false>> dwv{cuda::std::move(range), always<false>{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == buffer);
  }

  // multiple elements. all elements dropped
  {
    int buffer[] = {1, 2, 3, 4, 5};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), always<true>> dwv{cuda::std::move(range), always<true>{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == buffer + 5);
  }

  // multiple elements. some elements dropped
  {
    struct LessThan3
    {
      __host__ __device__ constexpr bool operator()(int i) const
      {
        return i < 3;
      }
    };
    int buffer[] = {1, 2, 3, 2, 1};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), LessThan3> dwv{cuda::std::move(range), LessThan3{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == buffer + 2);
  }

  // Make sure we do not make a copy of the predicate when we call begin()
  {
    int buffer[] = {1, 2, 3, 2, 1};
    bool moved = false, copied = false;
    auto range = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), TrackingPred> dwv{
      cuda::std::move(range), TrackingPred(&moved, &copied)};
    moved   = false;
    copied  = false;
    auto it = dwv.begin();
    assert(!moved);
    assert(!copied);
    unused(it);
  }

  // Test with a non-const predicate
  {
    auto mutable_pred = [](int& i) mutable {
      return i < 3;
    };
    int buffer[] = {1, 2, 3, 2, 1};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    // With MSVC the lambda does not satisfy indirect_unary_predicate
    if constexpr (cuda::std::indirect_unary_predicate<const decltype(mutable_pred), Iter>)
    {
      cuda::std::ranges::drop_while_view<decltype(range), decltype(mutable_pred)> dwv{
        cuda::std::move(range), mutable_pred};
      decltype(auto) it = dwv.begin();
      static_assert(cuda::std::same_as<decltype(it), Iter>);
      assert(base(it) == buffer + 2);
    }
    unused(range);
    unused(mutable_pred);
  }

  // Test with a predicate that takes by non-const reference
  {
    struct LessThan3
    {
      __host__ __device__ constexpr bool operator()(int& i) const
      {
        return i < 3;
      }
    };
    int buffer[] = {1, 2, 3, 2, 1};
    auto range   = make_subrange<Range, Iter, Sent>(buffer);
    cuda::std::ranges::drop_while_view<decltype(range), LessThan3> dwv{cuda::std::move(range), LessThan3{}};
    decltype(auto) it = dwv.begin();
    static_assert(cuda::std::same_as<decltype(it), Iter>);
    assert(base(it) == buffer + 2);
  }

  if constexpr (cuda::std::forward_iterator<Iter>)
  {
    // Make sure that we cache the result of begin() on subsequent calls
    {
      int buffer[] = {1, 2, 3, 2, 1};
      auto range   = make_subrange<Range, Iter, Sent>(buffer);

      int called = 0;
      auto pred  = [&](int i) {
        ++called;
        return i < 3;
      };
      cuda::std::ranges::drop_while_view<decltype(range), decltype(pred)> dwv{range, pred};
      for (auto i = 0; i < 10; ++i)
      {
        decltype(auto) it = dwv.begin();
        static_assert(cuda::std::same_as<decltype(it), Iter>);
        assert(base(it) == buffer + 2);
        assert(called == 3);
      }
    }
  }
}

__host__ __device__ constexpr bool test()
{
  testOne<cpp17_input_iterator<int*>>();
  testOne<cpp20_input_iterator<int*>>();
  testOne<forward_iterator<int*>>();
  testOne<bidirectional_iterator<int*>>();
  testOne<random_access_iterator<int*>>();
  testOne<contiguous_iterator<int*>>();
  testOne<int*>();
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
