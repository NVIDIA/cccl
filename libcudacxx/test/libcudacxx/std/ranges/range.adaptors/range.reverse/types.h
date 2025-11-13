//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_REVERSE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_REVERSE_TYPES_H

#include "test_iterators.h"
#include "test_macros.h"

struct BidirRange : cuda::std::ranges::view_base
{
  int* begin_;
  int* end_;

  __host__ __device__ constexpr BidirRange(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}

  __host__ __device__ constexpr bidirectional_iterator<int*> begin()
  {
    return bidirectional_iterator<int*>{begin_};
  }
  __host__ __device__ constexpr bidirectional_iterator<const int*> begin() const
  {
    return bidirectional_iterator<const int*>{begin_};
  }
  __host__ __device__ constexpr bidirectional_iterator<int*> end()
  {
    return bidirectional_iterator<int*>{end_};
  }
  __host__ __device__ constexpr bidirectional_iterator<const int*> end() const
  {
    return bidirectional_iterator<const int*>{end_};
  }
};
static_assert(cuda::std::ranges::bidirectional_range<BidirRange>);
static_assert(cuda::std::ranges::common_range<BidirRange>);
static_assert(cuda::std::ranges::view<BidirRange>);
static_assert(cuda::std::copyable<BidirRange>);

enum CopyCategory
{
  MoveOnly,
  Copyable
};
template <CopyCategory CC>
struct BidirSentRangeBase
{
  int* begin_;
  int* end_;

  __host__ __device__ constexpr BidirSentRangeBase(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  constexpr BidirSentRangeBase(const BidirSentRangeBase&)            = default;
  constexpr BidirSentRangeBase& operator=(const BidirSentRangeBase&) = default;
};

template <>
struct BidirSentRangeBase<MoveOnly>
{
  int* begin_;
  int* end_;

  __host__ __device__ constexpr BidirSentRangeBase(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  constexpr BidirSentRangeBase(BidirSentRangeBase&&)            = default;
  constexpr BidirSentRangeBase& operator=(BidirSentRangeBase&&) = default;
};

template <CopyCategory CC>
struct BidirSentRange
    : cuda::std::ranges::view_base
    , BidirSentRangeBase<CC>
{
  using sent_t       = sentinel_wrapper<bidirectional_iterator<int*>>;
  using sent_const_t = sentinel_wrapper<bidirectional_iterator<const int*>>;

  using Base = BidirSentRangeBase<CC>;
  using Base::Base;

  __host__ __device__ constexpr bidirectional_iterator<int*> begin()
  {
    return bidirectional_iterator<int*>{this->begin_};
  }
  __host__ __device__ constexpr bidirectional_iterator<const int*> begin() const
  {
    return bidirectional_iterator<const int*>{this->begin_};
  }
  __host__ __device__ constexpr sent_t end()
  {
    return sent_t{bidirectional_iterator<int*>{this->end_}};
  }
  __host__ __device__ constexpr sent_const_t end() const
  {
    return sent_const_t{bidirectional_iterator<const int*>{this->end_}};
  }
};
static_assert(cuda::std::ranges::bidirectional_range<BidirSentRange<MoveOnly>>);
static_assert(!cuda::std::ranges::common_range<BidirSentRange<MoveOnly>>);
static_assert(cuda::std::ranges::view<BidirSentRange<MoveOnly>>);
static_assert(!cuda::std::copyable<BidirSentRange<MoveOnly>>);
static_assert(cuda::std::ranges::bidirectional_range<BidirSentRange<Copyable>>);
static_assert(!cuda::std::ranges::common_range<BidirSentRange<Copyable>>);
static_assert(cuda::std::ranges::view<BidirSentRange<Copyable>>);
static_assert(cuda::std::copyable<BidirSentRange<Copyable>>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_REVERSE_TYPES_H
