//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ELEMENTS_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ELEMENTS_TYPES_H

#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

template <class T>
struct BufferView : cuda::std::ranges::view_base
{
  T* buffer_;
  cuda::std::size_t size_;

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferView(T (&b)[N])
      : buffer_(b)
      , size_(N)
  {}

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferView(cuda::std::array<T, N>& arr)
      : buffer_(arr.data())
      , size_(N)
  {}
};
using TupleBufferView = BufferView<cuda::std::tuple<int>>;

#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
#  define DELEGATE_TUPLEBUFFERVIEW(Derived)             \
    template <class T>                                  \
    __host__ __device__ constexpr Derived(T&& input)    \
        : TupleBufferView(cuda::std::forward<T>(input)) \
    {}
#else // ^^^ TEST_COMPILER_NVRTC ^^^ / vvv !TEST_COMPILER_NVRTC vvv
#  define DELEGATE_TUPLEBUFFERVIEW(Derived) using TupleBufferView::TupleBufferView;
#endif // !TEST_COMPILER_NVRTC

template <bool Simple>
struct Common : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(Common)

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr cuda::std::tuple<int>* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr const cuda::std::tuple<int>* begin() const
  {
    return buffer_;
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr cuda::std::tuple<int>* end()
  {
    return buffer_ + size_;
  }
  __host__ __device__ constexpr const cuda::std::tuple<int>* end() const
  {
    return buffer_ + size_;
  }
};
using SimpleCommon    = Common<true>;
using NonSimpleCommon = Common<false>;

using SimpleCommonRandomAccessSized    = SimpleCommon;
using NonSimpleCommonRandomAccessSized = NonSimpleCommon;

static_assert(cuda::std::ranges::common_range<Common<true>>);
static_assert(cuda::std::ranges::random_access_range<SimpleCommon>);
static_assert(cuda::std::ranges::sized_range<SimpleCommon>);
static_assert(simple_view<SimpleCommon>);
static_assert(!simple_view<NonSimpleCommon>);

template <bool Simple>
struct NonCommon : TupleBufferView
{
  DELEGATE_TUPLEBUFFERVIEW(NonCommon)

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr cuda::std::tuple<int>* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr const cuda::std::tuple<int>* begin() const
  {
    return buffer_;
  }

  template <bool Simple2 = Simple, cuda::std::enable_if_t<!Simple2, int> = 0>
  __host__ __device__ constexpr sentinel_wrapper<cuda::std::tuple<int>*> end()
  {
    return sentinel_wrapper<cuda::std::tuple<int>*>(buffer_ + size_);
  }
  __host__ __device__ constexpr sentinel_wrapper<const cuda::std::tuple<int>*> end() const
  {
    return sentinel_wrapper<const cuda::std::tuple<int>*>(buffer_ + size_);
  }
};

using SimpleNonCommon    = NonCommon<true>;
using NonSimpleNonCommon = NonCommon<false>;

static_assert(!cuda::std::ranges::common_range<SimpleNonCommon>);
static_assert(cuda::std::ranges::random_access_range<SimpleNonCommon>);
static_assert(!cuda::std::ranges::sized_range<SimpleNonCommon>);
static_assert(simple_view<SimpleNonCommon>);
static_assert(!simple_view<NonSimpleNonCommon>);

template <class Derived>
struct IterBase
{
  using iterator_concept = cuda::std::random_access_iterator_tag;
  using value_type       = cuda::std::tuple<int>;
  using difference_type  = intptr_t;

  __host__ __device__ constexpr cuda::std::tuple<int> operator*() const
  {
    return cuda::std::tuple<int>(5);
  }

  __host__ __device__ constexpr Derived& operator++()
  {
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) {}

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(const IterBase&, const IterBase&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const IterBase&, const IterBase&) noexcept
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(const IterBase&, const IterBase&) noexcept
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ELEMENTS_TYPES_H
