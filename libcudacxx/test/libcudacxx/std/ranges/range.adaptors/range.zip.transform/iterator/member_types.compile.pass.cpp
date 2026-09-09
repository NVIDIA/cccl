//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Iterator traits and member typedefs in zip_transform_view::iterator.

#include <cuda/std/array>
#include <cuda/std/ranges>

#include "../types.h"
#include "test_iterators.h"

template <class T>
_CCCL_CONCEPT HasIterCategory = _CCCL_REQUIRES_EXPR((T))(typename(typename T::iterator_category));

template <class T>
struct DiffTypeIter
{
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = T;

  TEST_FUNC int operator*() const;
  TEST_FUNC DiffTypeIter& operator++();
  TEST_FUNC void operator++(int);
  TEST_FUNC friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
};

template <class T>
struct DiffTypeRange
{
  TEST_FUNC DiffTypeIter<T> begin() const;
  TEST_FUNC DiffTypeIter<T> end() const;
};

struct Foo
{};
struct Bar
{};

struct RValueRefFn
{
  TEST_FUNC int&& operator()(auto&&...) const;
};

TEST_FUNC void test()
{
  int buffer[] = {1, 2, 3, 4};
  {
    // C++20 random_access C++17 random_access
    cuda::std::ranges::zip_transform_view v(GetFirst{}, buffer);
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 random_access C++17 input
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, buffer);
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 bidirectional C++17 bidirectional
    cuda::std::ranges::zip_transform_view v(GetFirst{}, BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 bidirectional C++17 input
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 forward C++17 bidirectional
    cuda::std::ranges::zip_transform_view v(GetFirst{}, ForwardSizedView{buffer});
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 forward C++17 input
    cuda::std::ranges::zip_transform_view v(MakeTuple{}, ForwardSizedView{buffer});
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // C++20 input C++17 not a range
    cuda::std::ranges::zip_transform_view v(GetFirst{}, InputCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(!HasIterCategory<Iter>);
  }

  {
    // difference_type of one view
    cuda::std::ranges::zip_transform_view v{MakeTuple{}, DiffTypeRange<cuda::std::intptr_t>{}};
    using Iter = decltype(v.begin());
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::intptr_t>);
  }

  {
    // difference_type of multiple views should be the common type
    cuda::std::ranges::zip_transform_view v{
      MakeTuple{}, DiffTypeRange<cuda::std::intptr_t>{}, DiffTypeRange<cuda::std::ptrdiff_t>{}};
    using Iter = decltype(v.begin());
    static_assert(
      cuda::std::is_same_v<Iter::difference_type, cuda::std::common_type_t<cuda::std::intptr_t, cuda::std::ptrdiff_t>>);
  }

  const cuda::std::array foos{Foo{}};
  cuda::std::array bars{Bar{}, Bar{}};
  {
    // value_type of one view
    cuda::std::ranges::zip_transform_view v{MakeTuple{}, foos};
    using Iter = decltype(v.begin());
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<Foo>>);
  }

  {
    // value_type of multiple views with different value_type
    cuda::std::ranges::zip_transform_view v{MakeTuple{}, foos, bars};
    using Iter = decltype(v.begin());
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<Foo, Bar>>);
  }

  // LWG3798 Rvalue reference and iterator_category
  {
    cuda::std::ranges::zip_transform_view v(RValueRefFn{}, buffer);
    using Iter = decltype(v.begin());
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
  }
}
