//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_ZIP_ITERATOR_H
#define TEST_CUDA_ITERATOR_ZIP_ITERATOR_H

#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

struct PODIter
{
  int i; // deliberately uninitialised

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ constexpr int operator*() const
  {
    return i;
  }

  __host__ __device__ constexpr PODIter& operator++()
  {
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) {}

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(const PODIter&, const PODIter&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const PODIter& lhs, const PODIter& rhs)
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const PODIter& lhs, const PODIter& rhs)
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER <=2017
};

struct IterNotDefaultConstructible
{
  int i; // deliberately uninitialised

  __host__ __device__ constexpr IterNotDefaultConstructible(const int val) noexcept
      : i(val)
  {}

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ constexpr int operator*() const
  {
    return i;
  }

  __host__ __device__ constexpr IterNotDefaultConstructible& operator++()
  {
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) {}

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool
  operator==(const IterNotDefaultConstructible&, const IterNotDefaultConstructible&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool
  operator==(const IterNotDefaultConstructible& lhs, const IterNotDefaultConstructible& rhs)
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool
  operator!=(const IterNotDefaultConstructible& lhs, const IterNotDefaultConstructible& rhs)
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER <=2017
};

struct IterNotDefaultConstructibleSized
{
  int i; // deliberately uninitialised

  __host__ __device__ constexpr IterNotDefaultConstructibleSized(const int val) noexcept
      : i(val)
  {}

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;
  using pointer           = int*;
  using reference         = int&;

  __host__ __device__ constexpr int operator*() const
  {
    return i;
  }

  __host__ __device__ constexpr IterNotDefaultConstructibleSized& operator++()
  {
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) {}

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool
  operator==(const IterNotDefaultConstructibleSized&, const IterNotDefaultConstructibleSized&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool
  operator==(const IterNotDefaultConstructibleSized& lhs, const IterNotDefaultConstructibleSized& rhs)
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool
  operator!=(const IterNotDefaultConstructibleSized& lhs, const IterNotDefaultConstructibleSized& rhs)
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER <=2017

  __host__ __device__ friend constexpr difference_type
  operator-(const IterNotDefaultConstructibleSized& x, const IterNotDefaultConstructibleSized& y)
  {
    return x.i - y.i;
  }
};
static_assert(::cuda::std::__has_random_access_traversal<IterNotDefaultConstructibleSized>);

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
  __host__ __device__ constexpr forward_sized_iterator(Base it)
      : it_(it)
  {}

  __host__ __device__ constexpr reference operator*() const
  {
    return *it_;
  }

  __host__ __device__ constexpr forward_sized_iterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr forward_sized_iterator operator++(int)
  {
    return forward_sized_iterator(it_++);
  }

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool
  operator==(const forward_sized_iterator&, const forward_sized_iterator&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend constexpr bool operator!=(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ != y.it_;
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ friend constexpr difference_type
  operator-(const forward_sized_iterator& x, const forward_sized_iterator& y)
  {
    return x.it_ - y.it_;
  }
};
static_assert(cuda::std::forward_iterator<forward_sized_iterator<>>);
static_assert(cuda::std::sized_sentinel_for<forward_sized_iterator<>, forward_sized_iterator<>>);

namespace adltest
{
struct iter_move_swap_iterator
{
  cuda::std::reference_wrapper<int> iter_move_called_times;
  cuda::std::reference_wrapper<int> iter_swap_called_times;
  int i = 0;

  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ TEST_CONSTEXPR_CXX20
  iter_move_swap_iterator(int& move_called, int& swap_called, int val = 0) noexcept
      : iter_move_called_times(move_called)
      , iter_swap_called_times(swap_called)
      , i(val)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX20 int operator*() const
  {
    return i;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX20 iter_move_swap_iterator& operator++()
  {
    ++i;
    return *this;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 void operator++(int)
  {
    ++i;
  }

#if TEST_STD_VER >= 2020
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator==(const iter_move_swap_iterator& x, cuda::std::default_sentinel_t)
  {
    return x.i == 5;
  }
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator==(const iter_move_swap_iterator& x, cuda::std::default_sentinel_t)
  {
    return x.i == 5;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator==(cuda::std::default_sentinel_t, const iter_move_swap_iterator& x)
  {
    return x.i == 5;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator!=(const iter_move_swap_iterator& x, cuda::std::default_sentinel_t)
  {
    return x.i != 5;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 bool
  operator!=(cuda::std::default_sentinel_t, const iter_move_swap_iterator& x)
  {
    return x.i != 5;
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ friend TEST_CONSTEXPR_CXX20 int iter_move(iter_move_swap_iterator const& it)
  {
    ++it.iter_move_called_times;
    return it.i;
  }
  __host__ __device__ friend TEST_CONSTEXPR_CXX20 void
  iter_swap(iter_move_swap_iterator const& x, iter_move_swap_iterator const& y)
  {
    ++x.iter_swap_called_times;
    ++y.iter_swap_called_times;
  }
};
} // namespace adltest

// This is for testing that zip iterator never calls underlying iterator's >, >=, <=, !=.
// The spec indicates that zip iterator's >= is negating zip iterator's < instead of calling underlying iterator's >=.
// Declare all the operations >, >=, <= etc to make it satisfy random_access_iterator concept,
// but not define them. If the zip iterator's >,>=, <=, etc isn't implemented in the way defined by the standard
// but instead calling underlying iterator's >,>=,<=, we will get a linker error for the runtime tests and
// non-constant expression for the compile time tests.
struct LessThanIterator
{
  int* it_           = nullptr;
  LessThanIterator() = default;
  __host__ __device__ constexpr LessThanIterator(int* it)
      : it_(it)
  {}

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ constexpr int& operator*() const
  {
    return *it_;
  }
  __host__ __device__ constexpr int& operator[](difference_type n) const
  {
    return it_[n];
  }
  __host__ __device__ constexpr LessThanIterator& operator++()
  {
    ++it_;
    return *this;
  }
  __host__ __device__ constexpr LessThanIterator& operator--()
  {
    --it_;
    return *this;
  }
  __host__ __device__ constexpr LessThanIterator operator++(int)
  {
    return LessThanIterator(it_++);
  }
  __host__ __device__ constexpr LessThanIterator operator--(int)
  {
    return LessThanIterator(it_--);
  }

  __host__ __device__ constexpr LessThanIterator& operator+=(difference_type n)
  {
    it_ += n;
    return *this;
  }
  __host__ __device__ constexpr LessThanIterator& operator-=(difference_type n)
  {
    it_ -= n;
    return *this;
  }

  __host__ __device__ constexpr friend LessThanIterator operator+(LessThanIterator x, difference_type n)
  {
    x += n;
    return x;
  }
  __host__ __device__ constexpr friend LessThanIterator operator+(difference_type n, LessThanIterator x)
  {
    x += n;
    return x;
  }
  __host__ __device__ constexpr friend LessThanIterator operator-(LessThanIterator x, difference_type n)
  {
    x -= n;
    return x;
  }
  __host__ __device__ constexpr friend difference_type operator-(LessThanIterator x, LessThanIterator y)
  {
    return x.it_ - y.it_;
  }

  __host__ __device__ constexpr friend bool operator==(LessThanIterator const& x, LessThanIterator const& y)
  {
    return x.it_ == y.it_;
  }
  __host__ __device__ friend bool operator!=(LessThanIterator const& x, LessThanIterator const& y);

  __host__ __device__ constexpr friend bool operator<(LessThanIterator const& x, LessThanIterator const& y)
  {
    return x.it_ < y.it_;
  }
  __host__ __device__ friend bool operator<=(LessThanIterator const&, LessThanIterator const&);
  __host__ __device__ friend bool operator>(LessThanIterator const&, LessThanIterator const&);
  __host__ __device__ friend bool operator>=(LessThanIterator const&, LessThanIterator const&);
};
static_assert(cuda::std::random_access_iterator<LessThanIterator>);

#endif // TEST_CUDA_ITERATOR_ZIP_ITERATOR_H
