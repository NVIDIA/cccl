//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef LIBCXX_TEST_SUPPORT_TEST_RANGE_H
#define LIBCXX_TEST_SUPPORT_TEST_RANGE_H

#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "test_iterators.h"

#if TEST_STD_VER < 2017
#  error "test/support/test_range.h" can only be included in builds supporting ranges
#endif

struct sentinel
{
  template <class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I>, int> = 0>
  __host__ __device__ friend bool operator==(sentinel const&, I const&)
  {
    return true;
  }
  template <class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I>, int> = 0>
  __host__ __device__ friend bool operator==(I const&, sentinel const&)
  {
    return true;
  }
  template <class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I>, int> = 0>
  __host__ __device__ friend bool operator!=(sentinel const&, I const&)
  {
    return false;
  }
  template <class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I>, int> = 0>
  __host__ __device__ friend bool operator!=(I const&, sentinel const&)
  {
    return false;
  }
};

template <template <class...> class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I<int*>>, int> = 0>
struct test_range
{
  __host__ __device__ I<int*> begin();
  __host__ __device__ I<int const*> begin() const;
  __host__ __device__ sentinel end();
  __host__ __device__ sentinel end() const;
};

template <template <class...> class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I<int*>>, int> = 0>
struct test_non_const_range
{
  __host__ __device__ I<int*> begin();
  __host__ __device__ sentinel end();
};

template <template <class...> class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I<int*>>, int> = 0>
struct test_common_range
{
  __host__ __device__ I<int*> begin();
  __host__ __device__ I<int const*> begin() const;
  __host__ __device__ I<int*> end();
  __host__ __device__ I<int const*> end() const;
};

template <template <class...> class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I<int*>>, int> = 0>
struct test_non_const_common_range
{
  __host__ __device__ I<int*> begin();
  __host__ __device__ I<int*> end();
};

template <template <class...> class I, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<I<int*>>, int> = 0>
struct test_view : cuda::std::ranges::view_base
{
  __host__ __device__ I<int*> begin();
  __host__ __device__ I<int const*> begin() const;
  __host__ __device__ sentinel end();
  __host__ __device__ sentinel end() const;
};

struct BorrowedRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
  __host__ __device__ BorrowedRange(BorrowedRange&&) = delete;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowedRange> = true;

static_assert(!cuda::std::ranges::view<BorrowedRange>, "");
static_assert(cuda::std::ranges::borrowed_range<BorrowedRange>, "");

using BorrowedView = cuda::std::ranges::empty_view<int>;
static_assert(cuda::std::ranges::view<BorrowedView>, "");
static_assert(cuda::std::ranges::borrowed_range<BorrowedView>, "");

using NonBorrowedView = cuda::std::ranges::single_view<int>;
static_assert(cuda::std::ranges::view<NonBorrowedView>, "");
static_assert(!cuda::std::ranges::borrowed_range<NonBorrowedView>, "");

#endif // LIBCXX_TEST_SUPPORT_TEST_RANGE_H
