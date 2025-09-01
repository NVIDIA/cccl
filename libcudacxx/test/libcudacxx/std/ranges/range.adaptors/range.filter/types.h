//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_FILTER_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_FILTER_TYPES_H

#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"

struct TrackInitialization
{
  __host__ __device__ constexpr explicit TrackInitialization(bool* moved, bool* copied)
      : moved_(moved)
      , copied_(copied)
  {}
  __host__ __device__ constexpr TrackInitialization(TrackInitialization const& other)
      : moved_(other.moved_)
      , copied_(other.copied_)
  {
    *copied_ = true;
  }
  __host__ __device__ constexpr TrackInitialization(TrackInitialization&& other)
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
  __host__ __device__ constexpr bool operator()(T const&) const
  {
    return true;
  }
};

struct AlwaysFalse
{
  template <typename T>
  __host__ __device__ constexpr bool operator()(T const&) const
  {
    return false;
  }
};

template <class Iterator, class Sentinel>
struct minimal_view : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr explicit minimal_view(Iterator it, Sentinel sent)
      : it_(base(cuda::std::move(it)))
      , sent_(base(cuda::std::move(sent)))
  {}

  minimal_view(minimal_view&&)            = default;
  minimal_view& operator=(minimal_view&&) = default;

  __host__ __device__ constexpr Iterator begin() const
  {
    return Iterator(it_);
  }
  __host__ __device__ constexpr Sentinel end() const
  {
    return Sentinel(sent_);
  }

private:
  decltype(base(cuda::std::declval<Iterator>())) it_;
  decltype(base(cuda::std::declval<Sentinel>())) sent_;
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
  __host__ __device__ explicit constexpr NoexceptIterMoveInputIterator(int* it)
      : it_(it)
  {}

  __host__ __device__ friend constexpr decltype(auto)
  iter_move(const NoexceptIterMoveInputIterator& it) noexcept(IsNoexcept)
  {
    return cuda::std::ranges::iter_move(it.it_);
  }

  __host__ __device__ friend constexpr int* base(const NoexceptIterMoveInputIterator& i)
  {
    return i.it_;
  }

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr NoexceptIterMoveInputIterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr NoexceptIterMoveInputIterator operator++(int)
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
  __host__ __device__ explicit constexpr NoexceptIterSwapInputIterator(int* it)
      : it_(it)
  {}

  __host__ __device__ friend constexpr void
  iter_swap(const NoexceptIterSwapInputIterator& a, const NoexceptIterSwapInputIterator& b) noexcept(IsNoexcept)
  {
    return cuda::std::ranges::iter_swap(a.it_, b.it_);
  }

  __host__ __device__ friend constexpr int* base(const NoexceptIterSwapInputIterator& i)
  {
    return i.it_;
  }

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr NoexceptIterSwapInputIterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr NoexceptIterSwapInputIterator operator++(int)
  {
    NoexceptIterSwapInputIterator tmp(*this);
    ++(*this);
    return tmp;
  }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_FILTER_TYPES_H
