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

// cuda::std::ranges::filter_view<V>::<iterator>() requires default_initializable<iterator_t<V>> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iterator, bool IsNoexcept>
TEST_FUNC constexpr void test_default_constructible()
{
  // Make sure the iterator is default constructible when the underlying iterator is.
  using View           = minimal_view<Iterator, sentinel_wrapper<Iterator>>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
  FilterIterator iter1{};
  FilterIterator iter2;
  assert(iter1 == iter2);
  // GCC 7 simply gives the wrong answer here. No amount of cajoling, pleading, or
  // massaging the code ever got it to pass this static_assert()
#if !_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >=, 8, 0)
  static_assert(cuda::std::is_nothrow_default_constructible_v<FilterIterator> == IsNoexcept);
#endif
}

template <class Iterator>
TEST_FUNC constexpr void test_not_default_constructible()
{
  // Make sure the iterator is *not* default constructible when the underlying iterator isn't.
  using View           = minimal_view<Iterator, sentinel_wrapper<Iterator>>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
  static_assert(!cuda::std::is_default_constructible_v<FilterIterator>);
}

TEST_FUNC constexpr bool tests()
{
  test_not_default_constructible<cpp17_input_iterator<int*>>();
  test_not_default_constructible<cpp20_input_iterator<int*>>();
  test_default_constructible<forward_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<bidirectional_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<random_access_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<contiguous_iterator<int*>, /* noexcept */ false>();
  test_default_constructible<int*, /* noexcept */ true>();
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
