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

TEST_GLOBAL_VARIABLE int globalBuff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

struct MoveOnlyView : cuda::std::ranges::view_base
{
  int start_;
  int* ptr_;
  TEST_FUNC constexpr explicit MoveOnlyView(int* ptr = globalBuff, int start = 0)
      : start_(start)
      , ptr_(ptr)
  {}
  constexpr MoveOnlyView(MoveOnlyView&&)            = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  TEST_FUNC constexpr int* begin() const
  {
    return ptr_ + start_;
  }
  TEST_FUNC constexpr int* end() const
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
  TEST_FUNC constexpr explicit CopyableView(int start = 0)
      : start_(start)
  {}
  constexpr CopyableView(CopyableView const&)            = default;
  constexpr CopyableView& operator=(CopyableView const&) = default;
  TEST_FUNC constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  TEST_FUNC constexpr int* end() const
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
  TEST_FUNC constexpr explicit ForwardView(int* ptr = globalBuff)
      : ptr_(ptr)
  {}
  constexpr ForwardView(ForwardView&&)            = default;
  constexpr ForwardView& operator=(ForwardView&&) = default;
  TEST_FUNC constexpr auto begin() const
  {
    return forward_iterator<int*>(ptr_);
  }
  TEST_FUNC constexpr auto end() const
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
  TEST_FUNC RandomAccessIter begin() const noexcept;
  TEST_FUNC RandomAccessIter end() const noexcept;
};
static_assert(cuda::std::ranges::view<RandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<RandomAccessView>);

using BidirectionalIter = bidirectional_iterator<int*>;
struct BidirectionalView : cuda::std::ranges::view_base
{
  TEST_FUNC BidirectionalIter begin() const;
  TEST_FUNC BidirectionalIter end() const;
};
static_assert(cuda::std::ranges::view<BidirectionalView>);
static_assert(cuda::std::ranges::bidirectional_range<BidirectionalView>);

struct BorrowableRange
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableRange> = true;
static_assert(!cuda::std::ranges::view<BorrowableRange>);
static_assert(cuda::std::ranges::contiguous_range<BorrowableRange>);
static_assert(cuda::std::ranges::borrowed_range<BorrowableRange>);

struct InputView : cuda::std::ranges::view_base
{
  int* ptr_;
  TEST_FUNC constexpr explicit InputView(int* ptr = globalBuff)
      : ptr_(ptr)
  {}
  TEST_FUNC constexpr auto begin() const
  {
    return cpp20_input_iterator<int*>(ptr_);
  }
  TEST_FUNC constexpr auto end() const
  {
    return sentinel_wrapper<cpp20_input_iterator<int*>>(cpp20_input_iterator<int*>(ptr_ + 8));
  }
};
static_assert(cuda::std::ranges::view<InputView>);
static_assert(!cuda::std::ranges::sized_range<InputView>);

struct SizedSentinelView : cuda::std::ranges::view_base
{
  int count_;
  TEST_FUNC constexpr explicit SizedSentinelView(int count = 8)
      : count_(count)
  {}
  TEST_FUNC constexpr auto begin() const
  {
    return RandomAccessIter(globalBuff);
  }
  TEST_FUNC constexpr int* end() const
  {
    return globalBuff + count_;
  }
};
TEST_FUNC constexpr auto operator-(const RandomAccessIter& lhs, int* rhs)
{
  return base(lhs) - rhs;
}
TEST_FUNC constexpr auto operator-(int* lhs, const RandomAccessIter& rhs)
{
  return lhs - base(rhs);
}
TEST_FUNC constexpr bool operator==(const RandomAccessIter& lhs, int* rhs)
{
  return base(lhs) == rhs;
}
#if TEST_STD_VER <= 2017
TEST_FUNC constexpr bool operator==(int* lhs, const RandomAccessIter& rhs)
{
  return base(rhs) == lhs;
}
TEST_FUNC constexpr bool operator!=(const RandomAccessIter& lhs, int* rhs)
{
  return base(lhs) != rhs;
}
TEST_FUNC constexpr bool operator!=(int* lhs, const RandomAccessIter& rhs)
{
  return base(rhs) != lhs;
}
#endif // TEST_STD_VER <= 2017

struct SizedSentinelNotConstView : cuda::std::ranges::view_base
{
  TEST_FUNC ForwardIter begin() const;
  TEST_FUNC int* end() const;
  TEST_FUNC size_t size();
};
TEST_FUNC bool operator==(const ForwardIter& lhs, int* rhs);
#if TEST_STD_VER <= 2017
TEST_FUNC bool operator==(int* lhs, const ForwardIter& rhs);
TEST_FUNC bool operator!=(const ForwardIter& lhs, int* rhs);
TEST_FUNC bool operator!=(int* lhs, const ForwardIter& rhs);
#endif // TEST_STD_VER <= 2017

struct Range
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

using CountedIter = stride_counting_iterator<forward_iterator<int*>>;
struct CountedView : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr CountedIter begin() const
  {
    return CountedIter(ForwardIter(globalBuff));
  }
  TEST_FUNC constexpr CountedIter end() const
  {
    return CountedIter(ForwardIter(globalBuff + 8));
  }
};

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

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TRANSFORM_TYPES_H
