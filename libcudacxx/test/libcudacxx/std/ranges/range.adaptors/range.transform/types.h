//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

STATIC_TEST_GLOBAL_VAR int globalBuff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

struct MoveOnlyView : cuda::std::ranges::view_base
{
  int start_;
  int* ptr_;
  __host__ __device__ constexpr explicit MoveOnlyView(int* ptr = globalBuff, int start = 0)
      : start_(start)
      , ptr_(ptr)
  {}
  constexpr MoveOnlyView(MoveOnlyView&&)            = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  __host__ __device__ constexpr int* begin() const
  {
    return ptr_ + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return ptr_ + 8;
  }
};
static_assert(cuda::std::ranges::view<MoveOnlyView>);
static_assert(cuda::std::ranges::contiguous_range<MoveOnlyView>);
static_assert(!cuda::std::copyable<MoveOnlyView>);

struct CopyableView : cuda::std::ranges::view_base
{
  int start_;
  __host__ __device__ constexpr explicit CopyableView(int start = 0)
      : start_(start)
  {}
  constexpr CopyableView(CopyableView const&)            = default;
  constexpr CopyableView& operator=(CopyableView const&) = default;
  __host__ __device__ constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};
static_assert(cuda::std::ranges::view<CopyableView>);
static_assert(cuda::std::ranges::contiguous_range<CopyableView>);
static_assert(cuda::std::copyable<CopyableView>);

using ForwardIter = forward_iterator<int*>;
struct ForwardView : cuda::std::ranges::view_base
{
  int* ptr_;
  __host__ __device__ constexpr explicit ForwardView(int* ptr = globalBuff)
      : ptr_(ptr)
  {}
  constexpr ForwardView(ForwardView&&)            = default;
  constexpr ForwardView& operator=(ForwardView&&) = default;
  __host__ __device__ constexpr auto begin() const
  {
    return forward_iterator<int*>(ptr_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return forward_iterator<int*>(ptr_ + 8);
  }
};
static_assert(cuda::std::ranges::view<ForwardView>);
static_assert(cuda::std::ranges::forward_range<ForwardView>);

using ForwardRange = test_common_range<forward_iterator>;
static_assert(!cuda::std::ranges::view<ForwardRange>);
static_assert(cuda::std::ranges::forward_range<ForwardRange>);

using RandomAccessIter = random_access_iterator<int*>;
struct RandomAccessView : cuda::std::ranges::view_base
{
  __host__ __device__ RandomAccessIter begin() const noexcept;
  __host__ __device__ RandomAccessIter end() const noexcept;
};
static_assert(cuda::std::ranges::view<RandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<RandomAccessView>);

using BidirectionalIter = bidirectional_iterator<int*>;
struct BidirectionalView : cuda::std::ranges::view_base
{
  __host__ __device__ BidirectionalIter begin() const;
  __host__ __device__ BidirectionalIter end() const;
};
static_assert(cuda::std::ranges::view<BidirectionalView>);
static_assert(cuda::std::ranges::bidirectional_range<BidirectionalView>);

struct BorrowableRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableRange> = true;
static_assert(!cuda::std::ranges::view<BorrowableRange>);
static_assert(cuda::std::ranges::contiguous_range<BorrowableRange>);
static_assert(cuda::std::ranges::borrowed_range<BorrowableRange>);

struct InputView : cuda::std::ranges::view_base
{
  int* ptr_;
  __host__ __device__ constexpr explicit InputView(int* ptr = globalBuff)
      : ptr_(ptr)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return cpp20_input_iterator<int*>(ptr_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel_wrapper<cpp20_input_iterator<int*>>(cpp20_input_iterator<int*>(ptr_ + 8));
  }
};
static_assert(cuda::std::ranges::view<InputView>);
static_assert(!cuda::std::ranges::sized_range<InputView>);

struct SizedSentinelView : cuda::std::ranges::view_base
{
  int count_;
  __host__ __device__ constexpr explicit SizedSentinelView(int count = 8)
      : count_(count)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return RandomAccessIter(globalBuff);
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + count_;
  }
};
__host__ __device__ constexpr auto operator-(const RandomAccessIter& lhs, int* rhs)
{
  return base(lhs) - rhs;
}
__host__ __device__ constexpr auto operator-(int* lhs, const RandomAccessIter& rhs)
{
  return lhs - base(rhs);
}
__host__ __device__ constexpr bool operator==(const RandomAccessIter& lhs, int* rhs)
{
  return base(lhs) == rhs;
}
#if TEST_STD_VER <= 2017
__host__ __device__ constexpr bool operator==(int* lhs, const RandomAccessIter& rhs)
{
  return base(rhs) == lhs;
}
__host__ __device__ constexpr bool operator!=(const RandomAccessIter& lhs, int* rhs)
{
  return base(lhs) != rhs;
}
__host__ __device__ constexpr bool operator!=(int* lhs, const RandomAccessIter& rhs)
{
  return base(rhs) != lhs;
}
#endif // TEST_STD_VER <= 2017

struct SizedSentinelNotConstView : cuda::std::ranges::view_base
{
  __host__ __device__ ForwardIter begin() const;
  __host__ __device__ int* end() const;
  __host__ __device__ size_t size();
};
__host__ __device__ bool operator==(const ForwardIter& lhs, int* rhs);
#if TEST_STD_VER <= 2017
__host__ __device__ bool operator==(int* lhs, const ForwardIter& rhs);
__host__ __device__ bool operator!=(const ForwardIter& lhs, int* rhs);
__host__ __device__ bool operator!=(int* lhs, const ForwardIter& rhs);
#endif // TEST_STD_VER <= 2017

struct Range
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

using CountedIter = stride_counting_iterator<forward_iterator<int*>>;
struct CountedView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr CountedIter begin() const
  {
    return CountedIter(ForwardIter(globalBuff));
  }
  __host__ __device__ constexpr CountedIter end() const
  {
    return CountedIter(ForwardIter(globalBuff + 8));
  }
};

struct TimesTwo
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x * 2;
  }
};

struct PlusOneMutable
{
  __host__ __device__ constexpr int operator()(int x)
  {
    return x + 1;
  }
};

struct PlusOne
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

struct Increment
{
  __host__ __device__ constexpr int& operator()(int& x)
  {
    return ++x;
  }
};

struct IncrementRvalueRef
{
  __host__ __device__ constexpr int&& operator()(int& x)
  {
    return cuda::std::move(++x);
  }
};

struct PlusOneNoexcept
{
  __host__ __device__ constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H
