//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_FACTORIES_RANGE_IOTA_VIEW_TYPES_H
#define TEST_STD_RANGES_RANGE_FACTORIES_RANGE_IOTA_VIEW_TYPES_H

#include "test_macros.h"

struct SomeInt
{
  using difference_type = int;

  int value_;
  __host__ __device__ constexpr explicit SomeInt(int value = 0)
      : value_(value)
  {}

#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  auto operator<=>(const SomeInt&) const = default;
#else
  __host__ __device__ friend constexpr bool operator==(const SomeInt& lhs, const SomeInt& rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(const SomeInt& lhs, const SomeInt& rhs)
  {
    return lhs.value_ != rhs.value_;
  }

  __host__ __device__ friend constexpr bool operator<(const SomeInt& lhs, const SomeInt& rhs)
  {
    return lhs.value_ < rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator<=(const SomeInt& lhs, const SomeInt& rhs)
  {
    return lhs.value_ <= rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator>(const SomeInt& lhs, const SomeInt& rhs)
  {
    return lhs.value_ > rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator>=(const SomeInt& lhs, const SomeInt& rhs)
  {
    return lhs.value_ >= rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr SomeInt& operator+=(SomeInt& lhs, const SomeInt& rhs)
  {
    lhs.value_ += rhs.value_;
    return lhs;
  }
  __host__ __device__ friend constexpr SomeInt& operator-=(SomeInt& lhs, const SomeInt& rhs)
  {
    lhs.value_ -= rhs.value_;
    return lhs;
  }

  __host__ __device__ friend constexpr SomeInt& operator+=(SomeInt& lhs, difference_type rhs)
  {
    lhs.value_ += rhs;
    return lhs;
  }
  __host__ __device__ friend constexpr SomeInt& operator-=(SomeInt& lhs, difference_type rhs)
  {
    lhs.value_ -= rhs;
    return lhs;
  }

  __host__ __device__ friend constexpr SomeInt operator+(SomeInt lhs, SomeInt rhs)
  {
    return SomeInt{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr int operator-(SomeInt lhs, SomeInt rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ friend constexpr SomeInt operator+(SomeInt lhs, difference_type rhs)
  {
    return SomeInt{lhs.value_ + rhs};
  }
  __host__ __device__ friend constexpr int operator-(SomeInt lhs, difference_type rhs)
  {
    return lhs.value_ - rhs;
  }

  __host__ __device__ friend constexpr SomeInt operator+(difference_type lhs, SomeInt rhs)
  {
    return SomeInt{lhs + rhs.value_};
  }
  __host__ __device__ friend constexpr int operator-(difference_type lhs, SomeInt rhs)
  {
    return lhs - rhs.value_;
  }

  __host__ __device__ constexpr SomeInt& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr SomeInt operator++(int)
  {
    auto tmp = *this;
    ++value_;
    return tmp;
  }
  __host__ __device__ constexpr SomeInt& operator--()
  {
    --value_;
    return *this;
  }
  __host__ __device__ constexpr SomeInt operator--(int)
  {
    auto tmp = *this;
    --value_;
    return tmp;
  }
};

template <class T>
struct IntComparableWith
{
  using difference_type = cuda::std::iter_difference_t<T>;

  T value_;
  __host__ __device__ constexpr explicit IntComparableWith(T value = T())
      : value_(value)
  {}

  __host__ __device__ friend constexpr bool operator==(IntComparableWith lhs, IntComparableWith rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator==(IntComparableWith lhs, T rhs)
  {
    return lhs.value_ == rhs;
  }
  __host__ __device__ friend constexpr bool operator==(T lhs, IntComparableWith rhs)
  {
    return lhs == rhs.value_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(IntComparableWith lhs, IntComparableWith rhs)
  {
    return lhs.value_ != rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(IntComparableWith lhs, T rhs)
  {
    return lhs.value_ != rhs;
  }
  __host__ __device__ friend constexpr bool operator!=(T lhs, IntComparableWith rhs)
  {
    return lhs != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr IntComparableWith operator+(IntComparableWith lhs, IntComparableWith rhs)
  {
    return IntComparableWith{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr difference_type operator-(IntComparableWith lhs, IntComparableWith rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ constexpr IntComparableWith& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr IntComparableWith operator++(int)
  {
    auto tmp = *this;
    ++value_;
    return tmp;
  }
  __host__ __device__ constexpr IntComparableWith operator--()
  {
    --value_;
    return *this;
  }
};
template <class T>
__host__ __device__ IntComparableWith(T) -> IntComparableWith<T>;

template <class T>
struct IntSentinelWith
{
  using difference_type = cuda::std::iter_difference_t<T>;

  T value_;
  __host__ __device__ constexpr explicit IntSentinelWith(T value = T())
      : value_(value)
  {}

  __host__ __device__ friend constexpr bool operator==(IntSentinelWith lhs, IntSentinelWith rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator==(IntSentinelWith lhs, T rhs)
  {
    return lhs.value_ == rhs;
  }
  __host__ __device__ friend constexpr bool operator==(T lhs, IntSentinelWith rhs)
  {
    return lhs == rhs.value_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(IntSentinelWith lhs, IntSentinelWith rhs)
  {
    return lhs.value_ != rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(IntSentinelWith lhs, T rhs)
  {
    return lhs.value_ != rhs;
  }
  __host__ __device__ friend constexpr bool operator!=(T lhs, IntSentinelWith rhs)
  {
    return lhs != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr IntSentinelWith operator+(IntSentinelWith lhs, IntSentinelWith rhs)
  {
    return IntSentinelWith{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr difference_type operator-(IntSentinelWith lhs, IntSentinelWith rhs)
  {
    return lhs.value_ - rhs.value_;
  }
  __host__ __device__ friend constexpr difference_type operator-(IntSentinelWith lhs, T rhs)
  {
    return lhs.value_ - rhs;
  }
  __host__ __device__ friend constexpr difference_type operator-(T lhs, IntSentinelWith rhs)
  {
    return lhs - rhs.value_;
  }

  __host__ __device__ constexpr IntSentinelWith& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr IntSentinelWith operator++(int)
  {
    auto tmp = *this;
    ++value_;
    return tmp;
  }
  __host__ __device__ constexpr IntSentinelWith operator--()
  {
    --value_;
    return *this;
  }
};
template <class T>
__host__ __device__ IntSentinelWith(T) -> IntSentinelWith<T>;

struct NotIncrementable
{
  using difference_type = int;

  int value_;
  __host__ __device__ constexpr explicit NotIncrementable(int value = 0)
      : value_(value)
  {}

#if TEST_STD_VER >= 2020
  bool operator==(const NotIncrementable&) const = default;
#else
  __host__ __device__ friend constexpr bool operator==(const NotIncrementable& lhs, const NotIncrementable& rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(const NotIncrementable& lhs, const NotIncrementable& rhs)
  {
    return lhs.value_ != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr NotIncrementable& operator+=(NotIncrementable& lhs, const NotIncrementable& rhs)
  {
    lhs.value_ += rhs.value_;
    return lhs;
  }
  __host__ __device__ friend constexpr NotIncrementable& operator-=(NotIncrementable& lhs, const NotIncrementable& rhs)
  {
    lhs.value_ -= rhs.value_;
    return lhs;
  }

  __host__ __device__ friend constexpr NotIncrementable operator+(NotIncrementable lhs, NotIncrementable rhs)
  {
    return NotIncrementable{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr int operator-(NotIncrementable lhs, NotIncrementable rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ constexpr NotIncrementable& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++value_;
  }
  __host__ __device__ constexpr NotIncrementable& operator--()
  {
    --value_;
    return *this;
  }
};
static_assert(!cuda::std::incrementable<NotIncrementable>);

struct NotDecrementable
{
  using difference_type = int;

  int value_;
  __host__ __device__ constexpr explicit NotDecrementable(int value = 0)
      : value_(value)
  {}

#if TEST_STD_VER >= 2020
  bool operator==(const NotDecrementable&) const = default;
#else
  __host__ __device__ friend constexpr bool operator==(const NotDecrementable& lhs, const NotDecrementable& rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(const NotDecrementable& lhs, const NotDecrementable& rhs)
  {
    return lhs.value_ != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr NotDecrementable& operator+=(NotDecrementable& lhs, const NotDecrementable& rhs)
  {
    lhs.value_ += rhs.value_;
    return lhs;
  }
  __host__ __device__ friend constexpr NotDecrementable& operator-=(NotDecrementable& lhs, const NotDecrementable& rhs)
  {
    lhs.value_ -= rhs.value_;
    return lhs;
  }

  __host__ __device__ friend constexpr NotDecrementable operator+(NotDecrementable lhs, NotDecrementable rhs)
  {
    return NotDecrementable{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr int operator-(NotDecrementable lhs, NotDecrementable rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ constexpr NotDecrementable& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++value_;
  }
};

enum CtorKind
{
  DefaultTo42,
  ValueCtor
};

template <CtorKind CK>
struct Int42
{
  using difference_type = int;

  int value_;
  __host__ __device__ constexpr explicit Int42(int value)
      : value_(value)
  {}
  template <CtorKind CK2 = CK, cuda::std::enable_if_t<CK2 == DefaultTo42, int> = 0>
  __host__ __device__ constexpr explicit Int42()
      : value_(42)
  {}

#if TEST_STD_VER >= 2020
  bool operator==(const Int42&) const = default;
#else
  __host__ __device__ friend constexpr bool operator==(const Int42& lhs, const Int42& rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  __host__ __device__ friend constexpr bool operator!=(const Int42& lhs, const Int42& rhs)
  {
    return lhs.value_ != rhs.value_;
  }
#endif

  __host__ __device__ friend constexpr Int42& operator+=(Int42& lhs, const Int42& rhs)
  {
    lhs.value_ += rhs.value_;
    return lhs;
  }
  __host__ __device__ friend constexpr Int42& operator-=(Int42& lhs, const Int42& rhs)
  {
    lhs.value_ -= rhs.value_;
    return lhs;
  }

  __host__ __device__ friend constexpr Int42 operator+(Int42 lhs, Int42 rhs)
  {
    return Int42{lhs.value_ + rhs.value_};
  }
  __host__ __device__ friend constexpr int operator-(Int42 lhs, Int42 rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  __host__ __device__ constexpr Int42& operator++()
  {
    ++value_;
    return *this;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++value_;
  }
};

#endif // TEST_STD_RANGES_RANGE_FACTORIES_RANGE_IOTA_VIEW_TYPES_H
