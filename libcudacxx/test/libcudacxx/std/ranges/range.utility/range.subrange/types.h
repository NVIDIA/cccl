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

__device__ int globalBuff[8];

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
  typedef cuda::std::forward_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef MoveOnlyForwardIter self;

  int* base = nullptr;

  MoveOnlyForwardIter()                                 = default;
  MoveOnlyForwardIter(MoveOnlyForwardIter&&)            = default;
  MoveOnlyForwardIter& operator=(MoveOnlyForwardIter&&) = default;
  MoveOnlyForwardIter(MoveOnlyForwardIter const&)       = delete;
  __host__ __device__ constexpr MoveOnlyForwardIter(int* ptr)
      : base(ptr)
  {}

  __host__ __device__ friend bool operator==(const self&, const self&);
#if TEST_STD_VER < 2020
  __host__ __device__ friend bool operator!=(const self&, const self&);
#endif

  __host__ __device__ friend constexpr bool operator==(const self& lhs, int* rhs)
  {
    return lhs.base == rhs;
  }
#if TEST_STD_VER < 2020 || defined(TEST_COMPILER_CLANG) || defined(TEST_COMPILER_NVRTC) || defined(TEST_COMPILER_MSVC)
  __host__ __device__ friend constexpr bool operator==(int* rhs, const self& lhs)
  {
    return lhs.base == rhs;
  }
  __host__ __device__ friend constexpr bool operator!=(const self& lhs, int* rhs)
  {
    return lhs.base != rhs;
  }
  __host__ __device__ friend constexpr bool operator!=(int* rhs, const self& lhs)
  {
    return lhs.base != rhs;
  }
#endif

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);
  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ constexpr operator pointer() const
  {
    return base;
  }
};

struct SizedSentinelForwardIter
{
  typedef cuda::std::forward_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef cuda::std::make_unsigned_t<cuda::std::ptrdiff_t> udifference_type;
  typedef SizedSentinelForwardIter self;

  SizedSentinelForwardIter() = default;
  __host__ __device__ constexpr explicit SizedSentinelForwardIter(int* ptr, bool* minusWasCalled)
      : base_(ptr)
      , minusWasCalled_(minusWasCalled)
  {}

  __host__ __device__ friend constexpr bool operator==(const self& lhs, const self& rhs)
  {
    return lhs.base_ == rhs.base_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(const self& lhs, const self& rhs)
  {
    return lhs.base_ != rhs.base_;
  }
#endif

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);
  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ friend constexpr difference_type
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
static_assert(cuda::std::sized_sentinel_for<SizedSentinelForwardIter, SizedSentinelForwardIter>, "");

struct ConvertibleForwardIter
{
  typedef cuda::std::forward_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef ConvertibleForwardIter self;

  int* base_ = nullptr;

  constexpr ConvertibleForwardIter() = default;
  __host__ __device__ constexpr explicit ConvertibleForwardIter(int* ptr)
      : base_(ptr)
  {}

  __host__ __device__ friend bool operator==(const self&, const self&);
#if TEST_STD_VER < 2020
  __host__ __device__ friend bool operator!=(const self&, const self&);
#endif

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);
  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ constexpr operator pointer() const
  {
    return base_;
  }

  // Explicitly deleted so this doesn't model sized_sentinel_for.
  __host__ __device__ friend constexpr difference_type operator-(int*, self const&) = delete;
  __host__ __device__ friend constexpr difference_type operator-(self const&, int*) = delete;
};
using ConvertibleForwardSubrange =
  cuda::std::ranges::subrange<ConvertibleForwardIter, int*, cuda::std::ranges::subrange_kind::unsized>;
static_assert(cuda::std::is_convertible_v<ConvertibleForwardIter, int*>, "");

template <bool EnableConvertible>
struct ConditionallyConvertibleBase
{
  typedef cuda::std::forward_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef cuda::std::make_unsigned_t<cuda::std::ptrdiff_t> udifference_type;
  typedef ConditionallyConvertibleBase self;

  int* base_ = nullptr;

  constexpr ConditionallyConvertibleBase() = default;
  __host__ __device__ constexpr explicit ConditionallyConvertibleBase(int* ptr)
      : base_(ptr)
  {}

  __host__ __device__ constexpr int* base() const
  {
    return base_;
  }

#if TEST_STD_VER < 2020
  __host__ __device__ friend bool operator==(const self& lhs, const self& rhs)
  {
    return lhs.base_ == rhs.base_;
  }
  __host__ __device__ friend bool operator!=(const self& lhs, const self& rhs)
  {
    return lhs.base_ != rhs.base_;
  }
#else
  __host__ __device__ friend bool operator==(const self&, const self&) = default;
#endif
  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);
  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  template <bool E = EnableConvertible, cuda::std::enable_if_t<E, int> = 0>
  __host__ __device__ constexpr operator pointer() const
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
  __host__ __device__ constexpr ForwardIter begin() const
  {
    return ForwardIter(globalBuff);
  }
  __host__ __device__ constexpr ForwardIter end() const
  {
    return ForwardIter(globalBuff + 8);
  }
};

namespace cuda
{
namespace std
{
namespace ranges
{
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<ForwardBorrowedRange> = true;
}
} // namespace std
} // namespace cuda

struct ForwardRange
{
  __host__ __device__ ForwardIter begin() const;
  __host__ __device__ ForwardIter end() const;
};

struct ConvertibleForwardBorrowedRange
{
  __host__ __device__ constexpr ConvertibleForwardIter begin() const
  {
    return ConvertibleForwardIter(globalBuff);
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};

namespace cuda
{
namespace std
{
namespace ranges
{
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<ConvertibleForwardBorrowedRange> = true;
}
} // namespace std
} // namespace cuda

struct ForwardBorrowedRangeDifferentSentinel
{
  struct sentinel
  {
    int* value;
    __host__ __device__ friend bool operator==(sentinel s, ForwardIter i)
    {
      return s.value == base(i);
    }
#if TEST_STD_VER < 2020
    __host__ __device__ friend bool operator==(ForwardIter i, sentinel s)
    {
      return s.value == base(i);
    }
    __host__ __device__ friend bool operator!=(sentinel s, ForwardIter i)
    {
      return s.value != base(i);
    }
    __host__ __device__ friend bool operator!=(ForwardIter i, sentinel s)
    {
      return s.value != base(i);
    }
#endif
  };

  __host__ __device__ constexpr ForwardIter begin() const
  {
    return ForwardIter(globalBuff);
  }
  __host__ __device__ constexpr sentinel end() const
  {
    return sentinel{globalBuff + 8};
  }
};

namespace cuda
{
namespace std
{
namespace ranges
{
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<ForwardBorrowedRangeDifferentSentinel> = true;
}
} // namespace std
} // namespace cuda

using DifferentSentinelSubrange = cuda::std::ranges::
  subrange<ForwardIter, ForwardBorrowedRangeDifferentSentinel::sentinel, cuda::std::ranges::subrange_kind::unsized>;

struct DifferentSentinelWithSizeMember
{
  struct sentinel
  {
    int* value;
    __host__ __device__ friend bool operator==(sentinel s, ForwardIter i)
    {
      return s.value == base(i);
    }
#if TEST_STD_VER < 2020
    __host__ __device__ friend bool operator==(ForwardIter i, sentinel s)
    {
      return s.value == base(i);
    }
    __host__ __device__ friend bool operator!=(sentinel s, ForwardIter i)
    {
      return s.value != base(i);
    }
    __host__ __device__ friend bool operator!=(ForwardIter i, sentinel s)
    {
      return s.value != base(i);
    }
#endif
  };

  __host__ __device__ constexpr ForwardIter begin() const
  {
    return ForwardIter(globalBuff);
  }
  __host__ __device__ constexpr sentinel end() const
  {
    return sentinel{globalBuff + 8};
  }
  __host__ __device__ constexpr size_t size() const
  {
    return 8;
  }
};

namespace cuda
{
namespace std
{
namespace ranges
{
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<DifferentSentinelWithSizeMember> = true;
}
} // namespace std
} // namespace cuda

using DifferentSentinelWithSizeMemberSubrange = cuda::std::ranges::
  subrange<ForwardIter, DifferentSentinelWithSizeMember::sentinel, cuda::std::ranges::subrange_kind::unsized>;

#endif // LIBCXX_TEST_STD_RANGES_RANGE_UTILITY_RANGE_SUBRANGE_TYPES_H
