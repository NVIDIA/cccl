//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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

struct XYPoint
{
  int x;
  int y;
};

template <class T>
_CCCL_CONCEPT has_arrow = _CCCL_REQUIRES_EXPR((T), T t)(t->x);
static_assert(has_arrow<XYPoint*>); // test the test

struct WithArrowOperator
{
  using iterator_category = cuda::std::input_iterator_tag;
  using difference_type   = cuda::std::ptrdiff_t;
  using value_type        = XYPoint;

  TEST_FUNC constexpr explicit WithArrowOperator(XYPoint* p)
      : p_(p)
  {}
  TEST_FUNC constexpr XYPoint& operator*() const
  {
    return *p_;
  }
  TEST_FUNC constexpr XYPoint* operator->() const
  {
    return p_;
  } // has arrow
  TEST_FUNC constexpr WithArrowOperator& operator++()
  {
    ++p_;
    return *this;
  }
  TEST_FUNC constexpr WithArrowOperator operator++(int)
  {
    return WithArrowOperator(p_++);
  }

  TEST_FUNC friend constexpr XYPoint* base(WithArrowOperator const& i)
  {
    return i.p_;
  }
  XYPoint* p_;
};
static_assert(cuda::std::input_iterator<WithArrowOperator>);

struct WithNonCopyableIterator : cuda::std::ranges::view_base
{
  struct iterator
  {
    using iterator_category = cuda::std::input_iterator_tag;
    using difference_type   = cuda::std::ptrdiff_t;
    using value_type        = XYPoint;

    iterator(iterator const&) = delete; // not copyable
    TEST_FUNC iterator(iterator&&);
    TEST_FUNC iterator& operator=(iterator&&) noexcept;
    TEST_FUNC XYPoint& operator*() const;
    TEST_FUNC XYPoint* operator->() const;
    TEST_FUNC iterator& operator++();
    TEST_FUNC iterator operator++(int);

    // We need this to use XYPoint* as a sentinel type below. sentinel_wrapper
    // can't be used because this iterator is not copyable.
    TEST_FUNC friend bool operator==(iterator const&, XYPoint*);
#if TEST_STD_VER <= 2017
    TEST_FUNC friend bool operator!=(iterator const&, XYPoint*);
    TEST_FUNC friend bool operator==(XYPoint*, iterator const&);
    TEST_FUNC friend bool operator!=(XYPoint*, iterator const&);
#endif
  };

  TEST_FUNC iterator begin() const;
  TEST_FUNC XYPoint* end() const;
};
static_assert(cuda::std::ranges::input_range<WithNonCopyableIterator>);

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
  cuda::std::array<XYPoint, 5> array{{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  using View           = minimal_view<Iter, Sent>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;

  for (cuda::std::ptrdiff_t n = 0; n != 5; ++n)
  {
    FilterView view =
      make_filter_view<FilterView, View, Iter, Sent>(array.data(), array.data() + array.size(), AlwaysTrue{});
    FilterIterator const iter(view, Iter(array.data() + n));
    decltype(auto) result = iter.operator->();
    static_assert(cuda::std::same_as<Iter, decltype(result)>);
    assert(base(result) == array.data() + n);
    assert(iter->x == n);
    assert(iter->y == n);
  }
}

template <class It>
TEST_FUNC constexpr void check_no_arrow()
{
  using View           = minimal_view<It, sentinel_wrapper<It>>;
  using FilterView     = cuda::std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = cuda::std::ranges::iterator_t<FilterView>;
  static_assert(!has_arrow<FilterIterator>);
}

TEST_FUNC constexpr bool tests()
{
  test<WithArrowOperator>();
  test<XYPoint*>();
  test<XYPoint const*>();
  test<contiguous_iterator<XYPoint*>>();
  test<contiguous_iterator<XYPoint const*>>();

  // Make sure filter_view::iterator doesn't have operator-> if the
  // underlying iterator doesn't have one.
  {
    check_no_arrow<cpp17_input_iterator<XYPoint*>>();
    check_no_arrow<cpp20_input_iterator<XYPoint*>>();
    check_no_arrow<forward_iterator<XYPoint*>>();
    check_no_arrow<bidirectional_iterator<XYPoint*>>();
    check_no_arrow<random_access_iterator<XYPoint*>>();
    check_no_arrow<int*>();
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
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(tests());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
