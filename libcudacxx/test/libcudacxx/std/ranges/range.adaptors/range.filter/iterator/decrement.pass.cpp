//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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
  TEST_FUNC constexpr bool operator()(int y) const
  {
    return x == y;
  }
};

template <class T>
_CCCL_CONCEPT has_pre_decrement = _CCCL_REQUIRES_EXPR((T), T t)((--t));

template <class T>
_CCCL_CONCEPT has_post_decrement = _CCCL_REQUIRES_EXPR((T), T t)(t--);

template <class Iterator>
using FilterIteratorFor = cuda::std::ranges::iterator_t<
  cuda::std::ranges::filter_view<minimal_view<Iterator, sentinel_wrapper<Iterator>>, EqualTo>>;

// needs to be a standalone function for NVRTC
template <class FilterView, class View, class Iter, class Sent, class T, class U, class V>
TEST_FUNC constexpr FilterView make_filter_view(T begin, U end, V pred)
{
  View view{Iter(begin), Sent(Iter(end))};
  return FilterView(cuda::std::move(view), pred);
}

template <class Iter, class Sent = sentinel_wrapper<Iter>>
TEST_FUNC constexpr void test()
{
  using View           = minimal_view<Iter, Sent>;
  using FilterView     = cuda::std::ranges::filter_view<View, EqualTo>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  // Test with a single satisfied value
  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view =
      make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = cuda::std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.data() + array.size()); // test the test

    FilterIterator& result = --it;
    static_assert(cuda::std::same_as<FilterIterator&, decltype(--it)>);
    assert(&result == &it);
    assert(base(result.base()) == array.data() + 1);
  }

  // Test with more than one satisfied value
  {
    cuda::std::array<int, 6> array{0, 1, 2, 3, 1, 4};
    FilterView view =
      make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = cuda::std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.data() + array.size()); // test the test

    FilterIterator& result = --it;
    assert(&result == &it);
    assert(base(result.base()) == array.data() + 4);

    --it;
    assert(base(it.base()) == array.data() + 1);
  }

  // Test going forward and then backward on the same iterator
  {
    cuda::std::array<int, 10> array{0, 1, 2, 3, 1, 1, 4, 5, 1, 6};
    FilterView view =
      make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = view.begin();
    ++it;
    --it;
    assert(base(it.base()) == array.data() + 1);
    ++it;
    ++it;
    --it;
    assert(base(it.base()) == array.data() + 4);
    ++it;
    ++it;
    --it;
    assert(base(it.base()) == array.data() + 5);
    ++it;
    ++it;
    --it;
    assert(base(it.base()) == array.data() + 8);
  }

  // Test post-decrement
  {
    cuda::std::array<int, 6> array{0, 1, 2, 3, 1, 4};
    FilterView view =
      make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), EqualTo{1});
    FilterIterator it = cuda::std::ranges::next(view.begin(), view.end());
    assert(base(it.base()) == array.data() + array.size()); // test the test

    FilterIterator result = it--;
    static_assert(cuda::std::same_as<FilterIterator, decltype(it--)>);
    assert(base(result.base()) == array.data() + array.size());
    assert(base(it.base()) == array.data() + 4);

    result = it--;
    assert(base(result.base()) == array.data() + 4);
    assert(base(it.base()) == array.data() + 1);
  }
}

TEST_FUNC constexpr bool tests()
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
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(tests());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
