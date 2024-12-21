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

#if TEST_STD_VER >= 2020
template <class T>
concept has_equal = requires(T const& x, T const& y) {
  { x == y };
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool has_equal = false;

template <class T>
inline constexpr bool
  has_equal<T, cuda::std::void_t<decltype(cuda::std::declval<const T&>() == cuda::std::declval<const T&>())>> = true;
#endif // TEST_STD_VER <= 2017

template <class Iterator>
__host__ __device__ constexpr void test()
{
  using Sentinel       = sentinel_wrapper<Iterator>;
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  {
    cuda::std::array<int, 5> array{0, 1, 2, 3, 4};
    FilterView view       = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, AlwaysTrue{}};
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
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, AlwaysTrue{}};
    assert(!(view.begin() == view.end()));
  }
}

__host__ __device__ constexpr bool tests()
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
    static_assert(!has_equal<FilterIterator>);
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
