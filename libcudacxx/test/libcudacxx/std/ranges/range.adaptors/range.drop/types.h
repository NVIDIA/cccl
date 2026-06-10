//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H

#include "test_iterators.h"
#include "test_macros.h"

TEST_GLOBAL_VARIABLE int globalBuff[8];

template <class T>
struct drop_sentinel
{
  T* ptr_;
  int* num_of_sentinel_cmp_calls;

public:
  TEST_FUNC friend constexpr bool operator==(drop_sentinel const s, T* const ptr) noexcept
  {
    ++(*s.num_of_sentinel_cmp_calls);
    return {s.ptr_ == ptr};
  }
  TEST_FUNC friend constexpr bool operator==(T* const ptr, drop_sentinel const s) noexcept
  {
    ++(*s.num_of_sentinel_cmp_calls);
    return {s.ptr_ == ptr};
  }
  TEST_FUNC friend constexpr bool operator!=(drop_sentinel const s, T* const ptr) noexcept
  {
    return !(s == ptr);
  }
  TEST_FUNC friend constexpr bool operator!=(T* const ptr, drop_sentinel const s) noexcept
  {
    return !(s == ptr);
  }
};

template <bool IsSimple>
struct MaybeSimpleNonCommonView : cuda::std::ranges::view_base
{
  int start_;
  int* num_of_sentinel_cmp_calls;
  TEST_FUNC constexpr cuda::std::size_t size() const
  {
    return 8;
  }
  TEST_FUNC constexpr int* begin()
  {
    return globalBuff + start_;
  }
  TEST_FUNC constexpr cuda::std::conditional_t<IsSimple, int*, const int*> begin() const
  {
    return globalBuff + start_;
  }
  TEST_FUNC constexpr drop_sentinel<int> end()
  {
    return drop_sentinel<int>{globalBuff + size(), num_of_sentinel_cmp_calls};
  }
  TEST_FUNC constexpr auto end() const
  {
    return cuda::std::conditional_t<IsSimple, drop_sentinel<int>, drop_sentinel<const int>>{
      globalBuff + size(), num_of_sentinel_cmp_calls};
  }
};

struct MoveOnlyView : cuda::std::ranges::view_base
{
  int start_;
  TEST_FUNC constexpr explicit MoveOnlyView(int start = 0)
      : start_(start)
  {}
  constexpr MoveOnlyView(MoveOnlyView&&)            = default;
  constexpr MoveOnlyView& operator=(MoveOnlyView&&) = default;
  TEST_FUNC constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  TEST_FUNC constexpr int* end() const
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

using ForwardIter = forward_iterator<int*>;
struct ForwardView : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr explicit ForwardView() noexcept {};
  constexpr ForwardView(ForwardView&&)            = default;
  constexpr ForwardView& operator=(ForwardView&&) = default;
  TEST_FUNC constexpr forward_iterator<int*> begin() const
  {
    return forward_iterator<int*>(globalBuff);
  }
  TEST_FUNC constexpr forward_iterator<int*> end() const
  {
    return forward_iterator<int*>(globalBuff + 8);
  }
};

struct ForwardRange
{
  TEST_FUNC ForwardIter begin() const;
  TEST_FUNC ForwardIter end() const;
};

struct ThrowingDefaultCtorForwardView : cuda::std::ranges::view_base
{
  TEST_FUNC ThrowingDefaultCtorForwardView() noexcept(false);
  TEST_FUNC ForwardIter begin() const;
  TEST_FUNC ForwardIter end() const;
};

struct NoDefaultCtorForwardView : cuda::std::ranges::view_base
{
  NoDefaultCtorForwardView() = delete;
  TEST_FUNC ForwardIter begin() const;
  TEST_FUNC ForwardIter end() const;
};

struct BorrowableRange
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct BorrowableView : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableView> = true;

struct InputView : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr cpp20_input_iterator<int*> begin() const
  {
    return cpp20_input_iterator<int*>(globalBuff);
  }
  TEST_FUNC constexpr int* end() const
  {
    return globalBuff + 8;
  }
};
// TODO: remove these bogus operators
TEST_FUNC constexpr bool operator==(const cpp20_input_iterator<int*>& lhs, int* rhs)
{
  return base(lhs) == rhs;
}
TEST_FUNC constexpr bool operator==(int* lhs, const cpp20_input_iterator<int*>& rhs)
{
  return base(rhs) == lhs;
}
#if TEST_STD_VER < 2020
TEST_FUNC constexpr bool operator!=(const cpp20_input_iterator<int*>& lhs, int* rhs)
{
  return base(lhs) != rhs;
}
TEST_FUNC constexpr bool operator!=(int* lhs, const cpp20_input_iterator<int*>& rhs)
{
  return base(rhs) != lhs;
}
#endif

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

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_DROP_TYPES_H
