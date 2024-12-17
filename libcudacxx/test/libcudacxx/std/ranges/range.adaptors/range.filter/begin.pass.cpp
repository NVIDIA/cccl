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

// constexpr iterator begin();

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

struct Range : cuda::std::ranges::view_base
{
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  __host__ __device__ constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr Iterator begin() const
  {
    return Iterator(begin_);
  }
  __host__ __device__ constexpr Sentinel end() const
  {
    return Sentinel(Iterator(end_));
  }

private:
  int* begin_;
  int* end_;
};

// A range that isn't a forward_range, used to test filter_view
// when we don't cache the result of begin()
struct InputRange : cuda::std::ranges::view_base
{
  using Iterator = cpp17_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  __host__ __device__ constexpr explicit InputRange(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr Iterator begin() const
  {
    return Iterator(begin_);
  }
  __host__ __device__ constexpr Sentinel end() const
  {
    return Sentinel(Iterator(end_));
  }

private:
  int* begin_;
  int* end_;
};

struct TrackingPred : TrackInitialization
{
  using TrackInitialization::TrackInitialization;
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

struct always_true
{
  __host__ __device__ constexpr bool operator()(int) const noexcept
  {
    return true;
  }
};
struct always_false
{
  __host__ __device__ constexpr bool operator()(int) const noexcept
  {
    return false;
  }
};

struct equals_to
{
  const int expected_;
  __host__ __device__ constexpr equals_to(const int expected) noexcept
      : expected_(expected)
  {}

  __host__ __device__ constexpr bool operator()(const int val) const noexcept
  {
    return val == expected_;
  }
};

template <typename Range>
__host__ __device__ constexpr void general_tests()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of `.begin()`
  {
    Range range(buff, buff + 1);
    cuda::std::ranges::filter_view<Range, always_true> view(range, always_true{});
    using FilterIterator = cuda::std::ranges::iterator_t<decltype(view)>;
    ASSERT_SAME_TYPE(FilterIterator, decltype(view.begin()));
  }

  // begin() over an empty range
  {
    Range range(buff, buff);
    cuda::std::ranges::filter_view<Range, always_true> view(range, always_true{});
    auto it = view.begin();
    assert(base(it.base()) == buff);
    assert(it == view.end());
  }

  // begin() over a 1-element range
  {
    {
      Range range(buff, buff + 1);
      cuda::std::ranges::filter_view<Range, equals_to> view(range, equals_to{1});
      auto it = view.begin();
      assert(base(it.base()) == buff);
    }
    {
      Range range(buff, buff + 1);
      cuda::std::ranges::filter_view<Range, always_false> view(range, always_false{});
      auto it = view.begin();
      assert(base(it.base()) == buff + 1);
      assert(it == view.end());
    }
  }

  // begin() over a 2-element range
  {
    {
      Range range(buff, buff + 2);
      cuda::std::ranges::filter_view<Range, equals_to> view(range, equals_to{1});
      auto it = view.begin();
      assert(base(it.base()) == buff);
    }
    {
      Range range(buff, buff + 2);
      cuda::std::ranges::filter_view<Range, equals_to> view(range, equals_to{2});
      auto it = view.begin();
      assert(base(it.base()) == buff + 1);
    }
    {
      Range range(buff, buff + 2);
      cuda::std::ranges::filter_view<Range, always_false> view(range, always_false{});
      auto it = view.begin();
      assert(base(it.base()) == buff + 2);
      assert(it == view.end());
    }
  }

  // begin() over a N-element range
  {
    for (int k = 1; k != 8; ++k)
    {
      Range range(buff, buff + 8);
      cuda::std::ranges::filter_view<Range, equals_to> view(range, equals_to{k});
      auto it = view.begin();
      assert(base(it.base()) == buff + (k - 1));
    }
    {
      Range range(buff, buff + 8);
      cuda::std::ranges::filter_view<Range, always_false> view(range, always_false{});
      auto it = view.begin();
      assert(base(it.base()) == buff + 8);
      assert(it == view.end());
    }
  }

  // Make sure we do not make a copy of the predicate when we call begin()
  // (we should be passing it to ranges::find_if using cuda::std::ref)
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 2);
    cuda::std::ranges::filter_view<Range, TrackingPred> view(range, TrackingPred(&moved, &copied));
    moved                    = false;
    copied                   = false;
    [[maybe_unused]] auto it = view.begin();
    assert(!moved);
    assert(!copied);
  }

  // Test with a non-const predicate
  {
    Range range(buff, buff + 8);
    auto pred = [](int i) mutable {
      return i % 2 == 0;
    };
    cuda::std::ranges::filter_view<Range, decltype(pred)> view(range, pred);
    auto it = view.begin();
    assert(base(it.base()) == buff + 1);
  }

  // Test with a predicate that takes by non-const reference
  {
    Range range(buff, buff + 8);
    auto pred = [](int& i) {
      return i % 2 == 0;
    };
    cuda::std::ranges::filter_view<Range, decltype(pred)> view(range, pred);
    auto it = view.begin();
    assert(base(it.base()) == buff + 1);
  }
}

template <typename ForwardRange>
__host__ __device__ constexpr void cache_tests()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Make sure that we cache the result of begin() on subsequent calls
  // (only applies to forward_ranges)
  ForwardRange range(buff, buff + 8);
  int called = 0;
  auto pred  = [&](int i) {
    ++called;
    return i == 3;
  };

  cuda::std::ranges::filter_view<Range, decltype(pred)> view(range, pred);
  assert(called == 0);
  for (int k = 0; k != 3; ++k)
  {
    auto it = view.begin();
    assert(base(it.base()) == buff + 2);
    assert(called == 3);
  }
}

__host__ __device__ constexpr bool test()
{
  general_tests<Range>();
  general_tests<InputRange>(); // test when we don't cache the result
  cache_tests<Range>();
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
