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

// constexpr iterator_t<V> operator->() const
//    requires has-arrow<iterator_t<V>> && copyable<iterator_t<V>>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4285) // operator-> is recursive if applied using infix notation
#endif // TEST_COMPILER_MSVC

struct Point
{
  int x;
  int y;
};

#if TEST_STD_VER >= 2020
template <class T>
concept has_arrow = requires(T t) {
  { t->x };
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool has_arrow = false;

template <class T>
inline constexpr bool has_arrow<T, cuda::std::void_t<decltype(cuda::std::declval<T>()->x)>> = true;
#endif // TEST_STD_VER <= 2017
static_assert(has_arrow<Point*>); // test the test

struct WithArrowOperator
{
  using iterator_category = cuda::std::input_iterator_tag;
  using difference_type   = cuda::std::ptrdiff_t;
  using value_type        = Point;

  __host__ __device__ constexpr explicit WithArrowOperator(Point* p)
      : p_(p)
  {}
  __host__ __device__ constexpr Point& operator*() const
  {
    return *p_;
  }
  __host__ __device__ constexpr Point* operator->() const
  {
    return p_;
  } // has arrow
  __host__ __device__ constexpr WithArrowOperator& operator++()
  {
    ++p_;
    return *this;
  }
  __host__ __device__ constexpr WithArrowOperator operator++(int)
  {
    return WithArrowOperator(p_++);
  }

  __host__ __device__ friend constexpr Point* base(WithArrowOperator const& i)
  {
    return i.p_;
  }
  Point* p_;
};
static_assert(cuda::std::input_iterator<WithArrowOperator>);

struct WithNonCopyableIterator : cuda::std::ranges::view_base
{
  struct iterator
  {
    using iterator_category = cuda::std::input_iterator_tag;
    using difference_type   = cuda::std::ptrdiff_t;
    using value_type        = Point;

    iterator(iterator const&) = delete; // not copyable
    __host__ __device__ iterator(iterator&&);
    __host__ __device__ iterator& operator=(iterator&&);
    __host__ __device__ Point& operator*() const;
    __host__ __device__ iterator operator->() const;
    __host__ __device__ iterator& operator++();
    __host__ __device__ iterator operator++(int);

    // We need this to use Point* as a sentinel type below. sentinel_wrapper
    // can't be used because this iterator is not copyable.
    __host__ __device__ friend bool operator==(iterator const&, Point*);
#if TEST_STD_VER <= 2017
    __host__ __device__ friend bool operator==(Point*, iterator const&);
    __host__ __device__ friend bool operator!=(iterator const&, Point*);
    __host__ __device__ friend bool operator!=(Point*, iterator const&);
#endif // TEST_STD_VER <= 2017
  };

  __host__ __device__ iterator begin() const;
  __host__ __device__ Point* end() const;
};
static_assert(cuda::std::ranges::input_range<WithNonCopyableIterator>);

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
__host__ __device__ constexpr void test()
{
  cuda::std::array<Point, 5> array{{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  using View           = minimal_view<Iterator, Sentinel>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  for (cuda::std::ptrdiff_t n = 0; n != 5; ++n)
  {
    FilterView view = FilterView{View{Iterator(array.begin()), Sentinel(Iterator(array.end()))}, AlwaysTrue{}};
    FilterIterator const iter(view, Iterator(array.begin() + n));
    decltype(auto) result = iter.operator->();
    static_assert(cuda::std::same_as<decltype(result), Iterator>);
    assert(base(result) == array.begin() + n);
    assert(iter->x == n);
    assert(iter->y == n);
  }
}

template <class It>
__host__ __device__ constexpr void check_no_arrow(It)
{
  using View           = minimal_view<It, sentinel_wrapper<It>>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
  static_assert(!has_arrow<FilterIterator>);
};

__host__ __device__ constexpr bool tests()
{
  test<WithArrowOperator>();
  test<Point*>();
  test<Point const*>();
  test<contiguous_iterator<Point*>>();
  test<contiguous_iterator<Point const*>>();

  // Make sure filter_view::iterator doesn't have operator-> if the
  // underlying iterator doesn't have one.
  {
    check_no_arrow(cpp17_input_iterator<Point*>{nullptr});
    check_no_arrow(cpp20_input_iterator<Point*>{nullptr});
    check_no_arrow(forward_iterator<Point*>{});
    check_no_arrow(bidirectional_iterator<Point*>{});
    check_no_arrow(random_access_iterator<Point*>{});
    check_no_arrow(static_cast<int*>(nullptr));
  }

  // Make sure filter_view::iterator doesn't have operator-> if the
  // underlying iterator is not copyable.
  {
    using FilterView     = cuda::std::ranges::filter_view<WithNonCopyableIterator, AlwaysTrue>;
    using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
    static_assert(!has_arrow<FilterIterator>);
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
