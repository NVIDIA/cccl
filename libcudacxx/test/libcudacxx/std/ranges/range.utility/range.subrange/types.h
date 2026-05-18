//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_STD_RANGES_RANGE_UTILITY_RANGE_SUBRANGE_TYPES_H
#define LIBCXX_TEST_STD_RANGES_RANGE_UTILITY_RANGE_SUBRANGE_TYPES_H

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

TEST_GLOBAL_VARIABLE int globalBuff[8];

struct Empty
{};

using InputIter   = cpp17_input_iterator<int*>;
using ForwardIter = forward_iterator<int*>;
using BidirIter   = bidirectional_iterator<int*>;

using ForwardSubrange =
  cuda::std::ranges::subrange<ForwardIter, ForwardIter, cuda::std::ranges::subrange_kind::unsized>;
using SizedIntPtrSubrange = cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>;

struct MoveOnlyForwardIter
{
  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;
  using self              = MoveOnlyForwardIter;

  int* base = nullptr;

  MoveOnlyForwardIter()                                 = default;
  MoveOnlyForwardIter(MoveOnlyForwardIter&&)            = default;
  MoveOnlyForwardIter& operator=(MoveOnlyForwardIter&&) = default;
  MoveOnlyForwardIter(MoveOnlyForwardIter const&)       = delete;
  TEST_FUNC constexpr MoveOnlyForwardIter(int* ptr)
      : base(ptr)
  {}

  TEST_FUNC friend bool operator==(const self&, const self&);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(const self&, const self&);
#endif

  TEST_FUNC friend constexpr bool operator==(const self& lhs, int* rhs)
  {
    return lhs.base == rhs;
  }
#if TEST_STD_VER < 2020 || TEST_COMPILER(CLANG) || TEST_COMPILER(NVRTC) || TEST_COMPILER(MSVC)
  TEST_FUNC friend constexpr bool operator==(int* rhs, const self& lhs)
  {
    return lhs.base == rhs;
  }
  TEST_FUNC friend constexpr bool operator!=(const self& lhs, int* rhs)
  {
    return lhs.base != rhs;
  }
  TEST_FUNC friend constexpr bool operator!=(int* rhs, const self& lhs)
  {
    return lhs.base != rhs;
  }
#endif

  TEST_FUNC reference operator*() const;
  TEST_FUNC pointer operator->() const;
  TEST_FUNC self& operator++();
  TEST_FUNC self operator++(int);
  TEST_FUNC self& operator--();
  TEST_FUNC self operator--(int);

  TEST_FUNC constexpr operator pointer() const
  {
    return base;
  }
};

struct SizedSentinelForwardIter
{
  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;
  using udifference_type  = cuda::std::make_unsigned_t<cuda::std::ptrdiff_t>;
  using self              = SizedSentinelForwardIter;

  SizedSentinelForwardIter() = default;
  TEST_FUNC constexpr explicit SizedSentinelForwardIter(int* ptr, bool* minusWasCalled)
      : base_(ptr)
      , minusWasCalled_(minusWasCalled)
  {}

  TEST_FUNC friend constexpr bool operator==(const self& lhs, const self& rhs)
  {
    return lhs.base_ == rhs.base_;
  }
#if TEST_STD_VER < 2020
  TEST_FUNC friend constexpr bool operator!=(const self& lhs, const self& rhs)
  {
    return lhs.base_ != rhs.base_;
  }
#endif

  TEST_FUNC reference operator*() const;
  TEST_FUNC pointer operator->() const;
  TEST_FUNC self& operator++();
  TEST_FUNC self operator++(int);
  TEST_FUNC self& operator--();
  TEST_FUNC self operator--(int);

  TEST_FUNC friend constexpr difference_type
  operator-(SizedSentinelForwardIter const& a, SizedSentinelForwardIter const& b)
  {
    if (a.minusWasCalled_)
    {
      *a.minusWasCalled_ = true;
    }
    if (b.minusWasCalled_)
    {
      *b.minusWasCalled_ = true;
    }
    return a.base_ - b.base_;
  }

private:
  int* base_            = nullptr;
  bool* minusWasCalled_ = nullptr;
};
static_assert(cuda::std::sized_sentinel_for<SizedSentinelForwardIter, SizedSentinelForwardIter>);

struct ConvertibleForwardIter
{
  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;
  using self              = ConvertibleForwardIter;

  int* base_ = nullptr;

  constexpr ConvertibleForwardIter() = default;
  TEST_FUNC constexpr explicit ConvertibleForwardIter(int* ptr)
      : base_(ptr)
  {}

  TEST_FUNC friend bool operator==(const self&, const self&);
#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator!=(const self&, const self&);
#endif

  TEST_FUNC reference operator*() const;
  TEST_FUNC pointer operator->() const;
  TEST_FUNC self& operator++();
  TEST_FUNC self operator++(int);
  TEST_FUNC self& operator--();
  TEST_FUNC self operator--(int);

  TEST_FUNC constexpr operator pointer() const
  {
    return base_;
  }

  // Explicitly deleted so this doesn't model sized_sentinel_for.
  TEST_FUNC friend constexpr difference_type operator-(int*, self const&) = delete;
  TEST_FUNC friend constexpr difference_type operator-(self const&, int*) = delete;
};
using ConvertibleForwardSubrange =
  cuda::std::ranges::subrange<ConvertibleForwardIter, int*, cuda::std::ranges::subrange_kind::unsized>;
static_assert(cuda::std::is_convertible_v<ConvertibleForwardIter, int*>);

template <bool EnableConvertible>
struct ConditionallyConvertibleBase
{
  using iterator_category = cuda::std::forward_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;
  using udifference_type  = cuda::std::make_unsigned_t<cuda::std::ptrdiff_t>;
  using self              = ConditionallyConvertibleBase;

  int* base_ = nullptr;

  constexpr ConditionallyConvertibleBase() = default;
  TEST_FUNC constexpr explicit ConditionallyConvertibleBase(int* ptr)
      : base_(ptr)
  {}

  TEST_FUNC constexpr int* base() const
  {
    return base_;
  }

#if TEST_STD_VER < 2020
  TEST_FUNC friend bool operator==(const self& lhs, const self& rhs)
  {
    return lhs.base_ == rhs.base_;
  }
  TEST_FUNC friend bool operator!=(const self& lhs, const self& rhs)
  {
    return lhs.base_ != rhs.base_;
  }
#else
  TEST_FUNC friend bool operator==(const self&, const self&) = default;
#endif
  TEST_FUNC reference operator*() const;
  TEST_FUNC pointer operator->() const;
  TEST_FUNC self& operator++();
  TEST_FUNC self operator++(int);
  TEST_FUNC self& operator--();
  TEST_FUNC self operator--(int);

  template <bool E = EnableConvertible, cuda::std::enable_if_t<E, int> = 0>
  TEST_FUNC constexpr operator pointer() const
  {
    return base_;
  }
};
using ConditionallyConvertibleIter = ConditionallyConvertibleBase<false>;
using SizedSentinelForwardSubrange = cuda::std::ranges::
  subrange<ConditionallyConvertibleIter, ConditionallyConvertibleIter, cuda::std::ranges::subrange_kind::sized>;
using ConvertibleSizedSentinelForwardIter = ConditionallyConvertibleBase<true>;
using ConvertibleSizedSentinelForwardSubrange =
  cuda::std::ranges::subrange<ConvertibleSizedSentinelForwardIter, int*, cuda::std::ranges::subrange_kind::sized>;

struct ForwardBorrowedRange
{
  TEST_FUNC constexpr ForwardIter begin() const
  {
    return ForwardIter(globalBuff);
  }
  TEST_FUNC constexpr ForwardIter end() const
  {
    return ForwardIter(globalBuff + 8);
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<ForwardBorrowedRange> = true;
} // namespace cuda::std::ranges

struct ForwardRange
{
  TEST_FUNC ForwardIter begin() const;
  TEST_FUNC ForwardIter end() const;
};

struct ConvertibleForwardBorrowedRange
{
  TEST_FUNC constexpr ConvertibleForwardIter begin() const
  {
    return ConvertibleForwardIter(globalBuff);
  }
  TEST_FUNC constexpr int* end() const
  {
    return globalBuff + 8;
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<ConvertibleForwardBorrowedRange> = true;
} // namespace cuda::std::ranges

struct ForwardBorrowedRangeDifferentSentinel
{
  struct sentinel
  {
    int* value;
    TEST_FUNC friend bool operator==(sentinel s, ForwardIter i)
    {
      return s.value == base(i);
    }
#if TEST_STD_VER < 2020
    TEST_FUNC friend bool operator==(ForwardIter i, sentinel s)
    {
      return s.value == base(i);
    }
    TEST_FUNC friend bool operator!=(sentinel s, ForwardIter i)
    {
      return s.value != base(i);
    }
    TEST_FUNC friend bool operator!=(ForwardIter i, sentinel s)
    {
      return s.value != base(i);
    }
#endif
  };

  TEST_FUNC constexpr ForwardIter begin() const
  {
    return ForwardIter(globalBuff);
  }
  TEST_FUNC constexpr sentinel end() const
  {
    return sentinel{globalBuff + 8};
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<ForwardBorrowedRangeDifferentSentinel> = true;
} // namespace cuda::std::ranges

using DifferentSentinelSubrange = cuda::std::ranges::
  subrange<ForwardIter, ForwardBorrowedRangeDifferentSentinel::sentinel, cuda::std::ranges::subrange_kind::unsized>;

struct DifferentSentinelWithSizeMember
{
  struct sentinel
  {
    int* value;
    TEST_FUNC friend bool operator==(sentinel s, ForwardIter i)
    {
      return s.value == base(i);
    }
#if TEST_STD_VER < 2020
    TEST_FUNC friend bool operator==(ForwardIter i, sentinel s)
    {
      return s.value == base(i);
    }
    TEST_FUNC friend bool operator!=(sentinel s, ForwardIter i)
    {
      return s.value != base(i);
    }
    TEST_FUNC friend bool operator!=(ForwardIter i, sentinel s)
    {
      return s.value != base(i);
    }
#endif
  };

  TEST_FUNC constexpr ForwardIter begin() const
  {
    return ForwardIter(globalBuff);
  }
  TEST_FUNC constexpr sentinel end() const
  {
    return sentinel{globalBuff + 8};
  }
  TEST_FUNC constexpr size_t size() const
  {
    return 8;
  }
};

namespace cuda::std::ranges
{
template <>
inline constexpr bool enable_borrowed_range<DifferentSentinelWithSizeMember> = true;
} // namespace cuda::std::ranges

using DifferentSentinelWithSizeMemberSubrange = cuda::std::ranges::
  subrange<ForwardIter, DifferentSentinelWithSizeMember::sentinel, cuda::std::ranges::subrange_kind::unsized>;

#endif // LIBCXX_TEST_STD_RANGES_RANGE_UTILITY_RANGE_SUBRANGE_TYPES_H
