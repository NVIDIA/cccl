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

// template<range R>
//   constexpr range_difference_t<R> ranges::distance(R&& r);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

template <class It, class Sent>
__host__ __device__ constexpr void test_ordinary()
{
  struct R
  {
    mutable int a[3] = {1, 2, 3};
    __host__ __device__ constexpr It begin() const
    {
      return It(a);
    }
    __host__ __device__ constexpr Sent end() const
    {
      return Sent(It(a + 3));
    }
  };
  R r;
  assert(cuda::std::ranges::distance(r) == 3);
  assert(cuda::std::ranges::distance(static_cast<R&&>(r)) == 3);
  assert(cuda::std::ranges::distance(static_cast<const R&>(r)) == 3);
  assert(cuda::std::ranges::distance(static_cast<const R&&>(r)) == 3);
  ASSERT_SAME_TYPE(decltype(cuda::std::ranges::distance(r)), cuda::std::ranges::range_difference_t<R>);
}

__host__ __device__ constexpr bool test()
{
  {
    using R = int[3];
    int a[] = {1, 2, 3};
    assert(cuda::std::ranges::distance(static_cast<R&>(a)) == 3);
    assert(cuda::std::ranges::distance(static_cast<R&&>(a)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const R&>(a)) == 3);
    assert(cuda::std::ranges::distance(static_cast<const R&&>(a)) == 3);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::distance(a)), cuda::std::ptrdiff_t);
    ASSERT_SAME_TYPE(decltype(cuda::std::ranges::distance(a)), cuda::std::ranges::range_difference_t<R>);
  }
  {
    // Unsized range, non-copyable iterator type, rvalue-ref-qualified begin()
    using It   = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<cpp20_input_iterator<int*>>;
    using R    = cuda::std::ranges::subrange<It, Sent, cuda::std::ranges::subrange_kind::unsized>;

    int a[] = {1, 2, 3};
    auto r  = R(It(a), Sent(It(a + 3)));
    assert(cuda::std::ranges::distance(r) == 3);
    assert(cuda::std::ranges::distance(static_cast<R&&>(r)) == 3);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), const R&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), const R&&>);
  }
  {
    // Sized range (unsized sentinel type), non-copyable iterator type, rvalue-ref-qualified begin()
    using It   = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<cpp20_input_iterator<int*>>;
    using R    = cuda::std::ranges::subrange<It, Sent, cuda::std::ranges::subrange_kind::sized>;

    int a[] = {1, 2, 3};
    auto r  = R(It(a), Sent(It(a + 3)), 3);
    assert(cuda::std::ranges::distance(r) == 3);
    assert(cuda::std::ranges::distance(static_cast<R&&>(r)) == 3);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), const R&>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), const R&&>);
  }
  {
    // Sized range (sized sentinel type), non-copyable iterator type
    test_ordinary<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();
  }
  test_ordinary<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_ordinary<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_ordinary<cpp17_output_iterator<int*>, sentinel_wrapper<cpp17_output_iterator<int*>>>();
  test_ordinary<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_ordinary<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>();
  test_ordinary<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>();
  test_ordinary<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();
  test_ordinary<int*, sentinel_wrapper<int*>>();

  test_ordinary<cpp17_input_iterator<int*>, sized_sentinel<cpp17_input_iterator<int*>>>();
  test_ordinary<cpp20_input_iterator<int*>, sized_sentinel<cpp20_input_iterator<int*>>>();
  test_ordinary<cpp17_output_iterator<int*>, sized_sentinel<cpp17_output_iterator<int*>>>();
  test_ordinary<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>();
  test_ordinary<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>();
  test_ordinary<random_access_iterator<int*>, sized_sentinel<random_access_iterator<int*>>>();
  test_ordinary<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test_ordinary<int*, sized_sentinel<int*>>();
  test_ordinary<int*, int*>();

  // Calling it on a non-range isn't allowed.
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), int>);
  static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::ranges::distance), int*>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
