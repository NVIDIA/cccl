//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr explicit sentinel(filter_view&);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

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
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterSentinel = cuda::std::ranges::sentinel_t<FilterView>;

  cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
  FilterView view =
    make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), AlwaysTrue{});

  FilterSentinel sent(view);
  assert(base(base(sent.base())) == base(base(view.end().base())));

  static_assert(!cuda::std::is_constructible_v<FilterSentinel, FilterView const&>);
  static_assert(!cuda::std::is_constructible_v<FilterSentinel, FilterView>);
  static_assert(cuda::std::is_constructible_v<FilterSentinel, FilterView&>
                && !cuda::std::is_convertible_v<FilterView&, FilterSentinel>);
}

TEST_FUNC constexpr bool tests()
{
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
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
