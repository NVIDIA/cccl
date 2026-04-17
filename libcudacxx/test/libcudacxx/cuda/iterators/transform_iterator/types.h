//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_TRANSFORM_ITERATOR_TYPES_H
#define TEST_CUDA_TRANSFORM_ITERATOR_TYPES_H

#include <cuda/std/utility>

#include "test_macros.h"

struct TimesTwo
{
  TEST_FUNC constexpr int operator()(int x) const
  {
    return x * 2;
  }
};

struct PlusOneMutable
{
  TEST_FUNC constexpr int operator()(int x)
  {
    return x + 1;
  }
};

struct PlusOne
{
  TEST_FUNC constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

struct Increment
{
  TEST_FUNC constexpr int& operator()(int& x)
  {
    return ++x;
  }
};

struct IncrementRvalueRef
{
  TEST_FUNC constexpr int&& operator()(int& x)
  {
    return cuda::std::move(++x);
  }
};

struct PlusOneNoexcept
{
  TEST_FUNC constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

struct PlusWithMutableMember
{
  int val_ = 0;
  TEST_FUNC constexpr PlusWithMutableMember(const int val) noexcept
      : val_(val)
  {}
  TEST_FUNC constexpr int operator()(int x) noexcept
  {
    return x + val_++;
  }
};

template <class Base = int*>
struct forward_sized_iterator
{
  Base it_ = nullptr;

  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;
  using pointer           = Base;
  using reference         = decltype(*Base{});

  forward_sized_iterator() = default;
  TEST_FUNC constexpr forward_sized_iterator(Base it)
      : it_(it)
  {}

  TEST_FUNC constexpr reference operator*() const
  {
    return *it_;
  }

  TEST_FUNC constexpr forward_sized_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  TEST_FUNC constexpr forward_sized_iterator operator++(int)
  {
    return forward_sized_iterator(it_++);
  }

#if TEST_STD_VER >= 2020
  TEST_FUNC friend constexpr bool operator==(const forward_sized_iterator&, const forward_sized_iterator&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  TEST_FUNC friend constexpr bool operator==(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ == y.it_;
  }
  TEST_FUNC friend constexpr bool operator!=(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ != y.it_;
  }
#endif // TEST_STD_VER <= 2017

  TEST_FUNC friend constexpr difference_type
  operator-(const forward_sized_iterator& x, const forward_sized_iterator& y) noexcept
  {
    return x.it_ - y.it_;
  }
};
static_assert(cuda::std::forward_iterator<forward_sized_iterator<>>);
static_assert(cuda::std::sized_sentinel_for<forward_sized_iterator<>, forward_sized_iterator<>>);

#endif // TEST_CUDA_TRANSFORM_ITERATOR_TYPES_H
