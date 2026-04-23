//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_FILTER_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_FILTER_TYPES_H

#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"

struct TrackInitialization
{
  TEST_FUNC constexpr explicit TrackInitialization(bool* moved, bool* copied)
      : moved_(moved)
      , copied_(copied)
  {}
  TEST_FUNC constexpr TrackInitialization(TrackInitialization const& other)
      : moved_(other.moved_)
      , copied_(other.copied_)
  {
    *copied_ = true;
  }
  TEST_FUNC constexpr TrackInitialization(TrackInitialization&& other)
      : moved_(other.moved_)
      , copied_(other.copied_)
  {
    *moved_ = true;
  }
  TrackInitialization& operator=(TrackInitialization const&) = default;
  TrackInitialization& operator=(TrackInitialization&&)      = default;
  bool* moved_;
  bool* copied_;
};

struct AlwaysTrue
{
  template <typename T>
  TEST_FUNC constexpr bool operator()(T const&) const
  {
    return true;
  }
};

struct AlwaysFalse
{
  template <typename T>
  TEST_FUNC constexpr bool operator()(T const&) const
  {
    return false;
  }
};

template <class Iter, class Sent>
struct minimal_view : cuda::std::ranges::view_base
{
  constexpr minimal_view() = default;

  TEST_FUNC constexpr explicit minimal_view(Iter it, Sent sent)
      : it_(base(cuda::std::move(it)))
      , sent_(base(cuda::std::move(sent)))
  {}

  minimal_view(minimal_view&&)            = default;
  minimal_view& operator=(minimal_view&&) = default;

  TEST_FUNC constexpr Iter begin() const
  {
    return Iter(it_);
  }
  TEST_FUNC constexpr Sent end() const
  {
    return Sent(sent_);
  }

private:
  decltype(base(cuda::std::declval<Iter>())) it_;
  decltype(base(cuda::std::declval<Sent>())) sent_;
};

template <bool IsNoexcept>
class NoexceptIterMoveInputIterator
{
  int* it_;

public:
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = typename cuda::std::iterator_traits<int*>::difference_type;
  using pointer           = int*;
  using reference         = int&;

  NoexceptIterMoveInputIterator() = default;
  TEST_FUNC explicit constexpr NoexceptIterMoveInputIterator(int* it)
      : it_(it)
  {}

  TEST_FUNC friend constexpr decltype(auto) iter_move(const NoexceptIterMoveInputIterator& it) noexcept(IsNoexcept)
  {
    return cuda::std::ranges::iter_move(it.it_);
  }

  TEST_FUNC friend constexpr int* base(const NoexceptIterMoveInputIterator& i)
  {
    return i.it_;
  }

  TEST_FUNC constexpr reference operator*() const
  {
    return *it_;
  }
  TEST_FUNC constexpr NoexceptIterMoveInputIterator& operator++()
  {
    ++it_;
    return *this;
  }
  TEST_FUNC constexpr NoexceptIterMoveInputIterator operator++(int)
  {
    NoexceptIterMoveInputIterator tmp(*this);
    ++(*this);
    return tmp;
  }
};

template <bool IsNoexcept>
class NoexceptIterSwapInputIterator
{
  int* it_;

public:
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = typename cuda::std::iterator_traits<int*>::difference_type;
  using pointer           = int*;
  using reference         = int&;

  NoexceptIterSwapInputIterator() = default;
  TEST_FUNC explicit constexpr NoexceptIterSwapInputIterator(int* it)
      : it_(it)
  {}

  TEST_FUNC friend constexpr void
  iter_swap(const NoexceptIterSwapInputIterator& a, const NoexceptIterSwapInputIterator& b) noexcept(IsNoexcept)
  {
    cuda::std::ranges::iter_swap(a.it_, b.it_);
  }

  TEST_FUNC friend constexpr int* base(const NoexceptIterSwapInputIterator& i)
  {
    return i.it_;
  }

  TEST_FUNC constexpr reference operator*() const
  {
    return *it_;
  }
  TEST_FUNC constexpr NoexceptIterSwapInputIterator& operator++()
  {
    ++it_;
    return *this;
  }
  TEST_FUNC constexpr NoexceptIterSwapInputIterator operator++(int)
  {
    NoexceptIterSwapInputIterator tmp(*this);
    ++(*this);
    return tmp;
  }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_FILTER_TYPES_H
