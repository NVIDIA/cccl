//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// ranges::advance(it, n, sent)

#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <bool Count, typename It>
__host__ __device__ constexpr void
check_forward(int* first, int* last, cuda::std::iter_difference_t<It> n, int* expected)
{
  using Difference   = cuda::std::iter_difference_t<It>;
  Difference const M = (expected - first); // expected travel distance

  {
    It it(first);
    auto sent             = sentinel_wrapper(It(last));
    decltype(auto) result = cuda::std::ranges::advance(it, n, sent);
    static_assert(cuda::std::same_as<decltype(result), Difference>);
    assert(result == n - M);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count)
  {
    auto it   = stride_counting_iterator(It(first));
    auto sent = sentinel_wrapper(stride_counting_iterator(It(last)));
    (void) cuda::std::ranges::advance(it, n, sent);
    // We don't have a sized sentinel, so we have to increment one-by-one
    // regardless of the iterator category.
    assert(it.stride_count() == M);
    assert(it.stride_displacement() == M);
  }
}

template <typename It>
__host__ __device__ constexpr void
check_forward_sized_sentinel(int* first, int* last, cuda::std::iter_difference_t<It> n, int* expected)
{
  using Difference      = cuda::std::iter_difference_t<It>;
  Difference const size = (last - first);
  Difference const M    = (expected - first); // expected travel distance

  {
    It it(first);
    auto sent             = distance_apriori_sentinel(size);
    decltype(auto) result = cuda::std::ranges::advance(it, n, sent);
    static_assert(cuda::std::same_as<decltype(result), Difference>);
    assert(result == n - M);
    assert(base(it) == expected);
  }

  // Count operations
  {
    auto it   = stride_counting_iterator(It(first));
    auto sent = distance_apriori_sentinel(size);
    (void) cuda::std::ranges::advance(it, n, sent);
    if constexpr (cuda::std::random_access_iterator<It>)
    {
      assert(it.stride_count() <= 1);
      assert(it.stride_displacement() <= 1);
    }
    else
    {
      assert(it.stride_count() == M);
      assert(it.stride_displacement() == M);
    }
  }
}

template <typename It>
__host__ __device__ constexpr void
check_backward(int* first, int* last, cuda::std::iter_difference_t<It> n, int* expected)
{
  static_assert(cuda::std::random_access_iterator<It>, "This test doesn't support non random access iterators");
  using Difference   = cuda::std::iter_difference_t<It>;
  Difference const M = (expected - last); // expected travel distance (which is negative)

  {
    It it(last);
    It sent(first);
    decltype(auto) result = cuda::std::ranges::advance(it, n, sent);
    static_assert(cuda::std::same_as<decltype(result), Difference>);
    assert(result == n - M);
    assert(base(it) == expected);
  }

  // Count operations
  {
    auto it   = stride_counting_iterator(It(last));
    auto sent = stride_counting_iterator(It(first));
    (void) cuda::std::ranges::advance(it, n, sent);
    assert(it.stride_count() <= 1);
    assert(it.stride_displacement() <= 1);
  }
}

struct iota_iterator
{
  using difference_type = int;
  using value_type      = int;

  __host__ __device__ constexpr int operator*() const
  {
    return x;
  }
  __host__ __device__ constexpr iota_iterator& operator++()
  {
    ++x;
    return *this;
  }
  __host__ __device__ constexpr iota_iterator operator++(int)
  {
    ++x;
    return iota_iterator{x - 1};
  }
#if TEST_STD_VER > 2017
  constexpr bool operator==(const iota_iterator&) const = default;
#else
  __host__ __device__ constexpr bool operator==(const iota_iterator& other) const
  {
    return x == other.x;
  }
  __host__ __device__ constexpr bool operator!=(const iota_iterator& other) const
  {
    return x != other.x;
  }
#endif
  __host__ __device__ constexpr int operator-(const iota_iterator& that) const
  {
    return x - that.x;
  }
  __host__ __device__ constexpr iota_iterator& operator--()
  {
    --x;
    return *this;
  }
  __host__ __device__ constexpr iota_iterator operator--(int)
  {
    --x;
    return iota_iterator{x + 1};
  }

  int x;
};
static_assert(cuda::std::bidirectional_iterator<iota_iterator>);
static_assert(cuda::std::sized_sentinel_for<iota_iterator, iota_iterator>);

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // Basic functionality test: advance forward, bound has the same type
  {
    int* p = nullptr;
    p      = range + 5;
    assert(cuda::std::ranges::advance(p, 0, range + 7) == 0);
    assert(p == range + 5);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, 1, range + 7) == 0);
    assert(p == range + 6);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, 2, range + 7) == 0);
    assert(p == range + 7);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, 3, range + 7) == 1);
    assert(p == range + 7);
  }

  // Basic functionality test: advance forward, bound is not the same type and not assignable
  {
    int* p         = nullptr;
    using ConstPtr = const int*;
    p              = range + 5;
    assert(cuda::std::ranges::advance(p, 0, ConstPtr(range + 7)) == 0);
    assert(p == range + 5);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, 1, ConstPtr(range + 7)) == 0);
    assert(p == range + 6);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, 2, ConstPtr(range + 7)) == 0);
    assert(p == range + 7);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, 3, ConstPtr(range + 7)) == 1);
    assert(p == range + 7);
  }

  // Basic functionality test: advance forward, bound has different type but assignable
  {
    const int* pc = nullptr;
    pc            = range + 5;
    assert(cuda::std::ranges::advance(pc, 0, range + 7) == 0);
    assert(pc == range + 5);
    pc = range + 5;
    assert(cuda::std::ranges::advance(pc, 1, range + 7) == 0);
    assert(pc == range + 6);
    pc = range + 5;
    assert(cuda::std::ranges::advance(pc, 2, range + 7) == 0);
    assert(pc == range + 7);
    pc = range + 5;
    assert(cuda::std::ranges::advance(pc, 3, range + 7) == 1);
    assert(pc == range + 7);
  }

  // Basic functionality test: advance backward, bound has the same type
  // Note that we don't test advancing backward with a bound of a different type because that's UB
  {
    int* p = nullptr;
    p      = range + 5;
    assert(cuda::std::ranges::advance(p, 0, range + 3) == 0);
    assert(p == range + 5);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, -1, range + 3) == 0);
    assert(p == range + 4);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, -2, range + 3) == 0);
    assert(p == range + 3);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, -3, range + 3) == -1);
    assert(p == range + 3);
  }

  // Basic functionality test: advance backward with an array as a sentinel
  {
    int* p = nullptr;
    p      = range + 5;
    assert(cuda::std::ranges::advance(p, 0, range) == 0);
    assert(p == range + 5);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, -1, range) == 0);
    assert(p == range + 4);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, -5, range) == 0);
    assert(p == range);
    p = range + 5;
    assert(cuda::std::ranges::advance(p, -6, range) == -1);
    assert(p == range);
  }

  // Exhaustive checks for n and range sizes
  for (int size = 0; size != 10; ++size)
  {
    for (int n = 0; n != 20; ++n)
    {
      {
        int* expected = n > size ? range + size : range + n;
        check_forward<false, cpp17_input_iterator<int*>>(range, range + size, n, expected);
        check_forward<false, cpp20_input_iterator<int*>>(range, range + size, n, expected);
        check_forward<true, forward_iterator<int*>>(range, range + size, n, expected);
        check_forward<true, bidirectional_iterator<int*>>(range, range + size, n, expected);
        check_forward<true, random_access_iterator<int*>>(range, range + size, n, expected);
        check_forward<true, contiguous_iterator<int*>>(range, range + size, n, expected);
        check_forward<true, int*>(range, range + size, n, expected);

        check_forward_sized_sentinel<cpp17_input_iterator<int*>>(range, range + size, n, expected);
        check_forward_sized_sentinel<cpp20_input_iterator<int*>>(range, range + size, n, expected);
        check_forward_sized_sentinel<forward_iterator<int*>>(range, range + size, n, expected);
        check_forward_sized_sentinel<bidirectional_iterator<int*>>(range, range + size, n, expected);
        check_forward_sized_sentinel<random_access_iterator<int*>>(range, range + size, n, expected);
        check_forward_sized_sentinel<contiguous_iterator<int*>>(range, range + size, n, expected);
        check_forward_sized_sentinel<int*>(range, range + size, n, expected);
      }

      {
        // Note that we can only test ranges::advance with a negative n for iterators that
        // are sized sentinels for themselves, because ranges::advance is UB otherwise.
        // In particular, that excludes bidirectional_iterators since those are not sized sentinels.
        int* expected = n > size ? range : range + size - n;
        check_backward<random_access_iterator<int*>>(range, range + size, -n, expected);
        check_backward<contiguous_iterator<int*>>(range, range + size, -n, expected);
        check_backward<int*>(range, range + size, -n, expected);
      }
    }
  }

  // Regression-test that INT_MIN doesn't cause any undefined behavior
  {
    auto i = iota_iterator{+1};
    assert(cuda::std::ranges::advance(i, INT_MIN, iota_iterator{-2}) == INT_MIN + 3);
    assert(i == iota_iterator{-2});
    i = iota_iterator{+1};
    assert(cuda::std::ranges::advance(i, -2, iota_iterator{INT_MIN + 1}) == 0);
    assert(i == iota_iterator{-1});
    i = iota_iterator{+1};
    assert(cuda::std::ranges::advance(i, INT_MIN, iota_iterator{INT_MIN + 1}) == 0);
    assert(i == iota_iterator{INT_MIN + 1});
  }

  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
