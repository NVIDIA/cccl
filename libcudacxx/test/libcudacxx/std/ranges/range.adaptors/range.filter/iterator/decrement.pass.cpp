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

// constexpr iterator& operator--() requires bidirectional_range<V>;
// constexpr iterator operator--(int) requires bidirectional_range<V>;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

struct EqualTo
{
  int x;
  __host__ __device__ constexpr bool operator()(int y) const
  {
    return x == y;
  }
};

#if TEST_STD_VER >= 2020
template <class T>
concept has_pre_decrement = requires(T t) {
  { --t };
};

template <class T>
concept has_post_decrement = requires(T t) {
  { t-- };
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool has_pre_decrement = false;

template <class T>
inline constexpr bool has_pre_decrement<T, cuda::std::void_t<decltype(--cuda::std::declval<T>())>> = true;

template <class T, class = void>
inline constexpr bool has_post_decrement = false;

template <class T>
inline constexpr bool has_post_decrement<T, cuda::std::void_t<decltype(cuda::std::declval<T>()--)>> = true;
#endif // TEST_STD_VER <= 2017

template <class Iterator>
using FilterIteratorFor = cuda::std::ranges::iterator_t<
  cuda::std::ranges::filter_view<minimal_view<Iterator, sentinel_wrapper<Iterator>>, EqualTo>>;

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
__host__ __device__ constexpr void test()
{
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, EqualTo>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  // Test with a single satisfied value
  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view   = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};
    FilterIterator it = cuda::std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.end()); // test the test

    FilterIterator& result = --it;
    ASSERT_SAME_TYPE(FilterIterator&, decltype(--it));
    assert(&result == &it);
    assert(base(result.base()) == array.begin() + 1);
  }

  // Test with more than one satisfied value
  {
    cuda::std::array<int, 6> array{0, 1, 2, 3, 1, 4};
    FilterView view   = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};
    FilterIterator it = cuda::std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.end()); // test the test

    FilterIterator& result = --it;
    assert(&result == &it);
    assert(base(result.base()) == array.begin() + 4);

    --it;
    assert(base(it.base()) == array.begin() + 1);
  }

  // Test going forward and then backward on the same iterator
  {
    cuda::std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view   = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};
    FilterIterator it = view.begin();
    ++it;
    --it;
    assert(base(it.base()) == array.begin() + 1);
    ++it;
    ++it;
    --it;
    assert(base(it.base()) == array.begin() + 4);
    ++it;
    ++it;
    --it;
    assert(base(it.base()) == array.begin() + 5);
    ++it;
    ++it;
    --it;
    assert(base(it.base()) == array.begin() + 8);
  }

  // Test post-decrement
  {
    cuda::std::array<int, 6> array{0, 1, 2, 3, 1, 4};
    FilterView view   = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, EqualTo{1}};
    FilterIterator it = cuda::std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.end()); // test the test

    FilterIterator result = it--;
    ASSERT_SAME_TYPE(FilterIterator, decltype(it--));
    assert(base(result.base()) == array.end());
    assert(base(it.base()) == array.begin() + 4);

    result = it--;
    assert(base(result.base()) == array.begin() + 4);
    assert(base(it.base()) == array.begin() + 1);
  }
}

__host__ __device__ constexpr bool tests()
{
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  test<bidirectional_iterator<int const*>>();
  test<random_access_iterator<int const*>>();
  test<contiguous_iterator<int const*>>();
  test<int const*>();

  // Make sure `operator--` isn't provided for non bidirectional ranges
  {
    static_assert(!has_pre_decrement<FilterIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(!has_pre_decrement<FilterIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(!has_pre_decrement<FilterIteratorFor<forward_iterator<int*>>>);

    static_assert(!has_post_decrement<FilterIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(!has_post_decrement<FilterIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(!has_post_decrement<FilterIteratorFor<forward_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(tests(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
