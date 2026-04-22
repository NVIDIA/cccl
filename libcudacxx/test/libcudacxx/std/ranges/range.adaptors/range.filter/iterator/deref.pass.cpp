//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// (clang-14 || gcc-12 || msvc-19.39) in C++20 tries to erroneously instantiate a bunch of
// default constructors that don't exist because it evaluates the class initializers before
// considering the default constructors requirements clause. It's not possible to selectively
// disable them in this file like the others, so we just disable the compiler entirely.

// UNSUPPORTED: (clang-14 || gcc-12 || msvc-19.39) && c++20

// constexpr range_reference_t<V> operator*() const;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/ranges>
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

template <class Iter, class ValueType = int, class Sent = sentinel_wrapper<Iter>>
TEST_FUNC constexpr void test()
{
  using View           = minimal_view<Iter, Sent>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  cuda::std::array array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  FilterView view =
    make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), AlwaysTrue{});

  for (cuda::std::size_t n = 0; n != array.size(); ++n)
  {
    FilterIterator const iter(view, Iter(array.data() + n));
    ValueType& result = *iter;
    static_assert(cuda::std::same_as<ValueType&, decltype(*iter)>);
    assert(&result == array.data() + n);
  }
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

  test<cpp17_input_iterator<int const*>, int const>();
  test<cpp20_input_iterator<int const*>, int const>();
  test<forward_iterator<int const*>, int const>();
  test<bidirectional_iterator<int const*>, int const>();
  test<random_access_iterator<int const*>, int const>();
  test<contiguous_iterator<int const*>, int const>();
  test<int const*, int const>();
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
