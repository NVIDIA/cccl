//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
#define TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

struct PlusOne
{
  __host__ __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};

struct PlusOneMutable
{
  __host__ __device__ constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

struct PlusOneMayThrow
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

#if !TEST_COMPILER(NVRTC)
struct PlusOneHost
{
  constexpr PlusOneHost() noexcept {}
  constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
struct PlusOneDevice
{
  __device__ constexpr PlusOneDevice() noexcept {}
  __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // _CCCL_CUDA_COMPILATION()

struct NotDefaultConstructiblePlusOne
{
  __host__ __device__ constexpr NotDefaultConstructiblePlusOne(int) noexcept {}
  __host__ __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};

struct TimesTwo
{
  __host__ __device__ constexpr int operator()(int x) const noexcept
  {
    return x * 2;
  }
};

struct TimesTwoMayThrow
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x * 2;
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
  operator-(const forward_sized_iterator& x, const forward_sized_iterator& y) noexcept
  {
    return x.it_ - y.it_;
  }
};
static_assert(cuda::std::forward_iterator<forward_sized_iterator<>>);
static_assert(cuda::std::sized_sentinel_for<forward_sized_iterator<>, forward_sized_iterator<>>);

#endif // TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
