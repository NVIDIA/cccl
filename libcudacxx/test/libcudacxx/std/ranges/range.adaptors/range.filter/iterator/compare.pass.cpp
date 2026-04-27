//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr bool operator==(iterator const&, iterator const&)
//  requires equality_comparable<iterator_t<V>>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

// needs to be a standalone function for NVRTC
template <class FilterView, class View, class Iter, class Sent, class T, class U, class V>
TEST_FUNC constexpr FilterView make_filter_view(T begin, U end, V pred)
{
  View view{Iter(begin), Sent(Iter(end))};
  return FilterView(cuda::std::move(view), pred);
}

template <class Iterator>
TEST_FUNC constexpr void test()
{
  using Sentinel       = sentinel_wrapper<Iterator>;
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view =
      make_filter_view<FilterView, View, Iterator, Sentinel>(array.data(), array.data() + array.size(), AlwaysTrue{});
    FilterIterator it1    = view.begin();
    FilterIterator it2    = view.begin();
    decltype(auto) result = (it1 == it2);
    static_assert(cuda::std::same_as<decltype(result), bool>);
    assert(result);

    ++it1;
    assert(!(it1 == it2));
  }

  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view =
      make_filter_view<FilterView, View, Iterator, Sentinel>(array.data(), array.data() + array.size(), AlwaysTrue{});
    assert(!(view.begin() == view.end()));
  }
}

TEST_FUNC constexpr bool tests()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  test<cpp17_input_iterator<int const*>>();
  test<forward_iterator<int const*>>();
  test<bidirectional_iterator<int const*>>();
  test<random_access_iterator<int const*>>();
  test<contiguous_iterator<int const*>>();
  test<int const*>();

  // Make sure `operator==` isn't provided for non comparable iterators
  {
    using Iterator       = cpp20_input_iterator<int*>;
    using Sentinel       = sentinel_wrapper<Iterator>;
    using FilterView     = cuda::std::ranges::filter_view<minimal_view<Iterator, Sentinel>, AlwaysTrue>;
    using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
    static_assert(!cuda::std::__weakly_equality_comparable_with<FilterIterator, FilterIterator>);
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
