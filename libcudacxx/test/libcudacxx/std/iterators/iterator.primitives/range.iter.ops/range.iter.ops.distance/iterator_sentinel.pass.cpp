//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<class I, sentinel_for<I> S>
//   requires (!sized_sentinel_for<S, I>)
//     constexpr iter_difference_t<I> ranges::distance(I first, S last);
//
// template<class I, sized_sentinel_for<decay_t<I>> S>
//   constexpr iter_difference_t<I> ranges::distance(I&& first, S last); // TODO: update when LWG3664 is resolved

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

template <class It, class Sent>
__host__ __device__ constexpr void test_unsized()
{
  static_assert(cuda::std::sentinel_for<Sent, It> && !cuda::std::sized_sentinel_for<Sent, It>);
  int a[3] = {1, 2, 3};
  {
    It first  = It(a);
    auto last = Sent(It(a));
    assert(cuda::std::ranges::distance(first, last) == 0);
    assert(cuda::std::ranges::distance(It(a), last) == 0);
    assert(cuda::std::ranges::distance(first, Sent(It(a))) == 0);
    assert(cuda::std::ranges::distance(It(a), Sent(It(a))) == 0);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::distance(It(a), Sent(It(a)))), cuda::std::iter_difference_t<It>);
  }
  {
    It first  = It(a);
    auto last = Sent(It(a + 3));
    assert(cuda::std::ranges::distance(first, last) == 3);

    // Test all const/ref-qualifications of both operands.
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<const Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<const Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<const Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<const Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<const Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<const Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<Sent&&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<const Sent&>(last)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<const Sent&&>(last)) == 3);
  }
}

template <class It, class Sent>
__host__ __device__ constexpr void test_sized()
{
  static_assert(cuda::std::sized_sentinel_for<Sent, It>);
  int a[] = {1, 2, 3};
  {
    It first  = It(a + 3);
    auto last = Sent(It(a));
    assert(cuda::std::ranges::distance(first, last) == -3);

    // Test all const/ref-qualifications of both operands.
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<const Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&>(first), static_cast<const Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<const Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<It&&>(first), static_cast<const Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<const Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&>(first), static_cast<const Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<Sent&&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<const Sent&>(last)) == -3);
    assert(cuda::std::ranges::distance(static_cast<const It&&>(first), static_cast<const Sent&&>(last)) == -3);
  }
  {
    It first  = It(a);
    auto last = Sent(It(a));
    assert(cuda::std::ranges::distance(first, last) == 0);
    assert(cuda::std::ranges::distance(It(a), last) == 0);
    assert(cuda::std::ranges::distance(first, Sent(It(a))) == 0);
    assert(cuda::std::ranges::distance(It(a), Sent(It(a))) == 0);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::distance(It(a), Sent(It(a)))), cuda::std::iter_difference_t<It>);
  }
  {
    It first  = It(a);
    auto last = Sent(It(a + 3));
    assert(cuda::std::ranges::distance(first, last) == 3);
    assert(cuda::std::ranges::distance(It(a), last) == 3);
    assert(cuda::std::ranges::distance(first, Sent(It(a + 3))) == 3);
    assert(cuda::std::ranges::distance(It(a), Sent(It(a + 3))) == 3);
  }
}

struct StrideCounter
{
  int* it_;
  int* inc_;
  using value_type      = int;
  using difference_type = int;
  __host__ __device__ explicit StrideCounter();
  __host__ __device__ constexpr explicit StrideCounter(int* it, int* inc)
      : it_(it)
      , inc_(inc)
  {}
  __host__ __device__ constexpr auto& operator++()
  {
    ++it_;
    *inc_ += 1;
    return *this;
  }
  __host__ __device__ StrideCounter operator++(int);
  __host__ __device__ int& operator*() const;
  __host__ __device__ bool operator==(StrideCounter) const;
  __host__ __device__ bool operator!=(StrideCounter) const;
};
static_assert(cuda::std::forward_iterator<StrideCounter>);
static_assert(!cuda::std::sized_sentinel_for<StrideCounter, StrideCounter>);

struct SizedStrideCounter
{
  int* it_;
  int* minus_;
  using value_type = int;
  __host__ __device__ explicit SizedStrideCounter();
  __host__ __device__ constexpr explicit SizedStrideCounter(int* it, int* minus)
      : it_(it)
      , minus_(minus)
  {}
  __host__ __device__ SizedStrideCounter& operator++();
  __host__ __device__ SizedStrideCounter operator++(int);
  __host__ __device__ int& operator*() const;
  __host__ __device__ bool operator==(SizedStrideCounter) const;
  __host__ __device__ bool operator!=(SizedStrideCounter) const;
  __host__ __device__ constexpr int operator-(SizedStrideCounter rhs) const
  {
    *minus_ += 1;
    return static_cast<int>(it_ - rhs.it_);
  }
};
static_assert(cuda::std::forward_iterator<SizedStrideCounter>);
static_assert(cuda::std::sized_sentinel_for<SizedStrideCounter, SizedStrideCounter>);

__host__ __device__ constexpr void test_stride_counting()
{
  {
    int a[] = {1, 2, 3};
    int inc = 0;
    StrideCounter first(a, &inc);
    StrideCounter last(a + 3, nullptr);
    decltype(auto) result = cuda::std::ranges::distance(first, last);
    static_assert(cuda::std::same_as<decltype(result), int>);
    assert(result == 3);
    assert(inc == 3);
  }
  {
    int a[]   = {1, 2, 3};
    int minus = 0;
    SizedStrideCounter first(a, &minus);
    SizedStrideCounter last(a + 3, nullptr);
    decltype(auto) result = cuda::std::ranges::distance(first, last);
    static_assert(cuda::std::same_as<decltype(result), int>);
    assert(result == 3);
    assert(minus == 1);
  }
}

__host__ __device__ constexpr bool test()
{
  {
    int a[] = {1, 2, 3};
    assert(cuda::std::ranges::distance(a, a + 3) == 3);
    assert(cuda::std::ranges::distance(a, a) == 0);
    assert(cuda::std::ranges::distance(a + 3, a) == -3);
  }

  test_unsized<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_unsized<cpp17_output_iterator<int*>, sentinel_wrapper<cpp17_output_iterator<int*>>>();
  test_unsized<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_unsized<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>();
  test_unsized<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>();
  test_unsized<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();
  test_unsized<int*, sentinel_wrapper<int*>>();
  test_unsized<const int*, sentinel_wrapper<const int*>>();
  test_unsized<forward_iterator<int*>, forward_iterator<int*>>();
  test_unsized<bidirectional_iterator<int*>, bidirectional_iterator<int*>>();

  test_sized<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();
  test_sized<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();
  test_sized<cpp17_output_iterator<int*>, sized_sentinel<cpp17_output_iterator<int*>>>();
  test_sized<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>();
  test_sized<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>();
  test_sized<random_access_iterator<int*>, sized_sentinel<random_access_iterator<int*>>>();
  test_sized<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test_sized<int*, sized_sentinel<int*>>();
  test_sized<const int*, sized_sentinel<const int*>>();
  test_sized<int*, int*>();
  test_sized<int*, const int*>();
  test_sized<random_access_iterator<int*>, random_access_iterator<int*>>();
  test_sized<contiguous_iterator<int*>, contiguous_iterator<int*>>();

  {
    using It = cpp20_input_iterator<int*>; // non-copyable, thus not a sentinel for itself
    static_assert(!cuda::std::is_copy_constructible_v<It>);
    static_assert(!cuda::std::sentinel_for<It, It>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&, It&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&, It&&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&&, It&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&&, It&&>);
  }
  {
    using It   = cpp20_input_iterator<int*>; // non-copyable
    using Sent = sentinel_wrapper<It>; // not a sized sentinel
    static_assert(cuda::std::sentinel_for<Sent, It> && !cuda::std::sized_sentinel_for<Sent, It>);
    int a[]   = {1, 2, 3};
    Sent last = Sent(It(a + 3));
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&, Sent&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&, Sent&&>);
    assert(cuda::std::ranges::distance(It(a), last) == 3);
    assert(cuda::std::ranges::distance(It(a), Sent(It(a + 3))) == 3);
  }
  {
    using It = cpp17_input_iterator<int*>; // not a sentinel for itself
    static_assert(!cuda::std::sentinel_for<It, It>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&, It&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&, It&&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&&, It&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), It&&, It&&>);
  }

  // Calling it on a non-iterator or non-sentinel isn't allowed.
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), int, int>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), int*, int>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), int, int*>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), int*, char*>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
