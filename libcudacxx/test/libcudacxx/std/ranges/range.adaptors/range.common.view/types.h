//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_COMMON_VIEW_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_COMMON_VIEW_TYPES_H

#include <cuda/std/ranges>

#include "test_iterators.h"

struct DefaultConstructibleView : cuda::std::ranges::view_base
{
  int* begin_                         = nullptr;
  int* end_                           = nullptr;
  explicit DefaultConstructibleView() = default;
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(end_);
  }
};
static_assert(cuda::std::ranges::view<DefaultConstructibleView>);
static_assert(cuda::std::default_initializable<DefaultConstructibleView>);

struct MoveOnlyView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;
  __host__ __device__ constexpr explicit MoveOnlyView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  constexpr MoveOnlyView(MoveOnlyView&&)            = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(end_);
  }
};
static_assert(cuda::std::ranges::view<MoveOnlyView>);
static_assert(cuda::std::ranges::contiguous_range<MoveOnlyView>);
static_assert(!cuda::std::copyable<MoveOnlyView>);

struct CopyableView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;
  __host__ __device__ constexpr explicit CopyableView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(end_);
  }
};
static_assert(cuda::std::ranges::view<CopyableView>);
static_assert(cuda::std::copyable<CopyableView>);

using ForwardIter = forward_iterator<int*>;
struct SizedForwardView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;
  __host__ __device__ constexpr explicit SizedForwardView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return forward_iterator<int*>(begin_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<forward_iterator<int*>>(forward_iterator<int*>(end_));
  }
};
static_assert(cuda::std::ranges::view<SizedForwardView>);
static_assert(cuda::std::ranges::forward_range<SizedForwardView>);
static_assert(cuda::std::ranges::sized_range<SizedForwardView>);

using RandomAccessIter = random_access_iterator<int*>;
struct SizedRandomAccessView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;
  __host__ __device__ constexpr explicit SizedRandomAccessView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return random_access_iterator<int*>(begin_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<random_access_iterator<int*>>(random_access_iterator<int*>(end_));
  }
};
static_assert(cuda::std::ranges::view<SizedRandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<SizedRandomAccessView>);
static_assert(cuda::std::ranges::sized_range<SizedRandomAccessView>);

struct CommonView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;
  __host__ __device__ constexpr explicit CommonView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return end_;
  }
};
static_assert(cuda::std::ranges::view<CommonView>);
static_assert(cuda::std::ranges::common_range<CommonView>);

struct NonCommonView : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;
  __host__ __device__ constexpr explicit NonCommonView(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<int*>(end_);
  }
};
static_assert(cuda::std::ranges::view<NonCommonView>);
static_assert(!cuda::std::ranges::common_range<NonCommonView>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_COMMON_VIEW_TYPES_H
