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

// template<viewable_range R>
// using all_t = decltype(views::all(declval<R>()));

#include <cuda/std/ranges>

#include "test_iterators.h"

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Range
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct BorrowableRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableRange> = true;

template <class T, class = void>
constexpr bool HasAllT = false;

template <class T>
constexpr bool HasAllT<T, cuda::std::void_t<cuda::std::views::all_t<T>>> = true;

// When T is a view, returns decay-copy(T)
ASSERT_SAME_TYPE(cuda::std::views::all_t<View>, View);
ASSERT_SAME_TYPE(cuda::std::views::all_t<View&>, View);
ASSERT_SAME_TYPE(cuda::std::views::all_t<View&&>, View);
ASSERT_SAME_TYPE(cuda::std::views::all_t<const View>, View);
ASSERT_SAME_TYPE(cuda::std::views::all_t<const View&>, View);
ASSERT_SAME_TYPE(cuda::std::views::all_t<const View&&>, View);

// Otherwise, when T is a reference to a range, returns ref_view<T>
ASSERT_SAME_TYPE(cuda::std::views::all_t<Range&>, cuda::std::ranges::ref_view<Range>);
ASSERT_SAME_TYPE(cuda::std::views::all_t<const Range&>, cuda::std::ranges::ref_view<const Range>);
ASSERT_SAME_TYPE(cuda::std::views::all_t<BorrowableRange&>, cuda::std::ranges::ref_view<BorrowableRange>);
ASSERT_SAME_TYPE(cuda::std::views::all_t<const BorrowableRange&>, cuda::std::ranges::ref_view<const BorrowableRange>);

// Otherwise, returns owning_view<T>
ASSERT_SAME_TYPE(cuda::std::views::all_t<Range>, cuda::std::ranges::owning_view<Range>);
ASSERT_SAME_TYPE(cuda::std::views::all_t<Range&&>, cuda::std::ranges::owning_view<Range>);
static_assert(!HasAllT<const Range>, "");
static_assert(!HasAllT<const Range&&>, "");
ASSERT_SAME_TYPE(cuda::std::views::all_t<BorrowableRange>, cuda::std::ranges::owning_view<BorrowableRange>);
ASSERT_SAME_TYPE(cuda::std::views::all_t<BorrowableRange&&>, cuda::std::ranges::owning_view<BorrowableRange>);
static_assert(!HasAllT<const BorrowableRange>, "");
static_assert(!HasAllT<const BorrowableRange&&>, "");

int main(int, char**)
{
  return 0;
}
