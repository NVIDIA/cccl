//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H

#include "test_iterators.h"
#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR int globalBuff[8];

template <class T>
struct drop_sentinel
{
  T* ptr_;
  int* num_of_sentinel_cmp_calls;

public:
  __host__ __device__ friend constexpr bool operator==(drop_sentinel const s, T* const ptr) noexcept
  {
    ++(*s.num_of_sentinel_cmp_calls);
    return {s.ptr_ == ptr};
  }
  __host__ __device__ friend constexpr bool operator==(T* const ptr, drop_sentinel const s) noexcept
  {
    ++(*s.num_of_sentinel_cmp_calls);
    return {s.ptr_ == ptr};
  }
  __host__ __device__ friend constexpr bool operator!=(drop_sentinel const s, T* const ptr) noexcept
  {
    return !(s == ptr);
  }
  __host__ __device__ friend constexpr bool operator!=(T* const ptr, drop_sentinel const s) noexcept
  {
    return !(s == ptr);
  }
};

template <bool IsSimple>
struct MaybeSimpleNonCommonView : cuda::std::ranges::view_base
{
  int start_;
  int* num_of_sentinel_cmp_calls;
  __host__ __device__ constexpr cuda::std::size_t size() const
  {
    return 8;
  }
  __host__ __device__ constexpr int* begin()
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr cuda::std::conditional_t<IsSimple, int*, const int*> begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr drop_sentinel<int> end()
  {
    return drop_sentinel<int>{globalBuff + size(), num_of_sentinel_cmp_calls};
  }
  __host__ __device__ constexpr auto end() const
  {
    return cuda::std::conditional_t<IsSimple, drop_sentinel<int>, drop_sentinel<const int>>{
      globalBuff + size(), num_of_sentinel_cmp_calls};
  }
};

struct MoveOnlyView : cuda::std::ranges::view_base
{
  int start_;
  __host__ __device__ constexpr explicit MoveOnlyView(int start = 0)
      : start_(start)
  {}
  constexpr MoveOnlyView(MoveOnlyView&&)            = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  __host__ __device__ constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
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

using ForwardIter = forward_iterator<int*>;
struct ForwardView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr explicit ForwardView() noexcept {};
  constexpr ForwardView(ForwardView&&)            = default;
  constexpr ForwardView& operator=(ForwardView&&) = default;
  __host__ __device__ constexpr forward_iterator<int*> begin() const
  {
    return forward_iterator<int*>(globalBuff);
  }
  __host__ __device__ constexpr forward_iterator<int*> end() const
  {
    return forward_iterator<int*>(globalBuff + 8);
  }
};

struct ForwardRange
{
  __host__ __device__ ForwardIter begin() const;
  __host__ __device__ ForwardIter end() const;
};

struct ThrowingDefaultCtorForwardView : cuda::std::ranges::view_base
{
  __host__ __device__ ThrowingDefaultCtorForwardView() noexcept(false);
  __host__ __device__ ForwardIter begin() const;
  __host__ __device__ ForwardIter end() const;
};

struct NoDefaultCtorForwardView : cuda::std::ranges::view_base
{
  NoDefaultCtorForwardView() = delete;
  __host__ __device__ ForwardIter begin() const;
  __host__ __device__ ForwardIter end() const;
};

struct BorrowableRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct BorrowableView : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableView> = true;

struct InputView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr cpp20_input_iterator<int*> begin() const
  {
    return cpp20_input_iterator<int*>(globalBuff);
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};
// TODO: remove these bogus operators
__host__ __device__ constexpr bool operator==(const cpp20_input_iterator<int*>& lhs, int* rhs)
{
  return base(lhs) == rhs;
}
__host__ __device__ constexpr bool operator==(int* lhs, const cpp20_input_iterator<int*>& rhs)
{
  return base(rhs) == lhs;
}
#if TEST_STD_VER < 2020
__host__ __device__ constexpr bool operator!=(const cpp20_input_iterator<int*>& lhs, int* rhs)
{
  return base(lhs) != rhs;
}
__host__ __device__ constexpr bool operator!=(int* lhs, const cpp20_input_iterator<int*>& rhs)
{
  return base(rhs) != lhs;
}
#endif

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

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H
