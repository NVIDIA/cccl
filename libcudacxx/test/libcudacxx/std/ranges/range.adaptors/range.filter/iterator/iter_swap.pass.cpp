//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr void iter_swap(iterator const& x, iterator const& y)
//  noexcept(noexcept(ranges::iter_swap(x.current_, y.current_)))
//  requires(indirectly_swappable<iterator_t<V>>);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class It>
_CCCL_CONCEPT has_iter_swap = _CCCL_REQUIRES_EXPR((It), It it)(cuda::std::ranges::iter_swap(it, it));

struct IsEven
{
  TEST_FUNC constexpr bool operator()(int x) const
  {
    return x % 2 == 0;
  }
};

// needs to be a standalone function for NVRTC
template <class FilterView, class View, class Iter, class Sent, class T, class U, class V>
TEST_FUNC constexpr FilterView make_filter_view(T begin, U end, V pred)
{
  View view{Iter(begin), Sent(Iter(end))};
  return FilterView(cuda::std::move(view), pred);
}

template <class Iterator, bool IsNoexcept>
TEST_FUNC constexpr void test()
{
  using Sentinel       = sentinel_wrapper<Iterator>;
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, IsEven>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  {
    cuda::std::array<int, 5> array{1, 2, 1, 4, 1};
    FilterView view =
      make_filter_view<FilterView, View, Iterator, Sentinel>(array.data(), array.data() + array.size(), IsEven{});
    FilterIterator const it1 = view.begin();
    FilterIterator const it2 = cuda::std::ranges::next(view.begin());

    static_assert(cuda::std::is_same_v<decltype(iter_swap(it1, it2)), void>);
    static_assert(noexcept(iter_swap(it1, it2)) == IsNoexcept);

    assert(*it1 == 2 && *it2 == 4); // test the test
    iter_swap(it1, it2);
    assert(*it1 == 4);
    assert(*it2 == 2);
  }
}

TEST_FUNC constexpr bool tests()
{
  test<cpp17_input_iterator<int*>, /* noexcept */ false>();
  test<cpp20_input_iterator<int*>, /* noexcept */ false>();
  test<forward_iterator<int*>, /* noexcept */ false>();
  test<bidirectional_iterator<int*>, /* noexcept */ false>();
  test<random_access_iterator<int*>, /* noexcept */ false>();
  test<contiguous_iterator<int*>, /* noexcept */ false>();
  test<int*, /* noexcept */ true>();
  test<NoexceptIterSwapInputIterator<true>, /* noexcept */ true>();
  test<NoexceptIterSwapInputIterator<false>, /* noexcept */ false>();

  // Test that iter_swap requires the underlying iterator to be iter_swappable
  {
    using Iterator       = int const*;
    using View           = minimal_view<Iterator, Iterator>;
    using FilterView     = cuda::std::ranges::filter_view<View, IsEven>;
    using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
    static_assert(!cuda::std::indirectly_swappable<Iterator>);
    static_assert(!has_iter_swap<FilterIterator>);
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
