//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_PSTL_H
#define SUPPORT_TEST_PSTL_H

#include <iostream>

#include <testing.cuh>

#include "test_macros.h"

struct nontrivial_type
{
  using difference_type = int;

  int value_;

  nontrivial_type() = default;

  // Not explicit to test conversions between types
  TEST_FUNC constexpr nontrivial_type(int value)
      : value_(value)
  {}

#if TEST_HAS_SPACESHIP()
  auto operator<=>(const nontrivial_type&) const = default;
#else
  TEST_FUNC friend constexpr bool operator==(const nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    return lhs.value_ == rhs.value_;
  }
  TEST_FUNC friend constexpr bool operator!=(const nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    return lhs.value_ != rhs.value_;
  }

  TEST_FUNC friend constexpr bool operator<(const nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    return lhs.value_ < rhs.value_;
  }
  TEST_FUNC friend constexpr bool operator<=(const nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    return lhs.value_ <= rhs.value_;
  }
  TEST_FUNC friend constexpr bool operator>(const nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    return lhs.value_ > rhs.value_;
  }
  TEST_FUNC friend constexpr bool operator>=(const nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    return lhs.value_ >= rhs.value_;
  }
#endif

  TEST_FUNC friend constexpr nontrivial_type& operator+=(nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    lhs.value_ += rhs.value_;
    return lhs;
  }
  TEST_FUNC friend constexpr nontrivial_type& operator-=(nontrivial_type& lhs, const nontrivial_type& rhs)
  {
    lhs.value_ -= rhs.value_;
    return lhs;
  }

  TEST_FUNC friend constexpr nontrivial_type& operator+=(nontrivial_type& lhs, difference_type rhs)
  {
    lhs.value_ += rhs;
    return lhs;
  }
  TEST_FUNC friend constexpr nontrivial_type& operator-=(nontrivial_type& lhs, difference_type rhs)
  {
    lhs.value_ -= rhs;
    return lhs;
  }

  TEST_FUNC friend constexpr nontrivial_type operator+(nontrivial_type lhs, nontrivial_type rhs)
  {
    return nontrivial_type{lhs.value_ + rhs.value_};
  }
  TEST_FUNC friend constexpr int operator-(nontrivial_type lhs, nontrivial_type rhs)
  {
    return lhs.value_ - rhs.value_;
  }

  TEST_FUNC friend constexpr nontrivial_type operator+(nontrivial_type lhs, difference_type rhs)
  {
    return nontrivial_type{lhs.value_ + rhs};
  }
  TEST_FUNC friend constexpr int operator-(nontrivial_type lhs, difference_type rhs)
  {
    return lhs.value_ - rhs;
  }

  TEST_FUNC friend constexpr nontrivial_type operator+(difference_type lhs, nontrivial_type rhs)
  {
    return nontrivial_type{lhs + rhs.value_};
  }
  TEST_FUNC friend constexpr int operator-(difference_type lhs, nontrivial_type rhs)
  {
    return lhs - rhs.value_;
  }

  TEST_FUNC constexpr nontrivial_type& operator++()
  {
    ++value_;
    return *this;
  }
  TEST_FUNC constexpr nontrivial_type operator++(int)
  {
    auto tmp = *this;
    ++value_;
    return tmp;
  }
  TEST_FUNC constexpr nontrivial_type& operator--()
  {
    --value_;
    return *this;
  }
  TEST_FUNC constexpr nontrivial_type operator--(int)
  {
    auto tmp = *this;
    --value_;
    return tmp;
  }

  // Also support thrust::sequence
  TEST_FUNC friend constexpr nontrivial_type operator*(const nontrivial_type lhs, const size_t rhs) noexcept
  {
    return nontrivial_type{lhs.value_ * static_cast<int>(rhs)};
  }
};

template <class T>
struct cast_to
{
  TEST_DEVICE_FUNC T operator()(int value) const noexcept
  {
    return static_cast<T>(value);
  }
};

using all_types =
  c2h::type_list<uint16_t,
                 int32_t,
                 uint64_t,
#if _CCCL_HAS_INT128()
                 __uint128_t,
                 __int128_t,
#endif // _CCCL_HAS_INT128()
                 float,
                 double,
#if _LIBCUDACXX_HAS_NVFP16()
                 __half,
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
                 __nv_bfloat16,
#endif // _LIBCUDACXX_HAS_NVBF16()
                 nontrivial_type>;

using arithmetic_types =
  c2h::type_list<uint16_t,
                 int32_t,
                 uint64_t,
#if _CCCL_HAS_INT128()
                 __uint128_t,
                 __int128_t,
#endif // _CCCL_HAS_INT128()
                 float,
                 double,
#if _LIBCUDACXX_HAS_NVFP16()
                 __half,
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
                 __nv_bfloat16
#endif // _LIBCUDACXX_HAS_NVBF16()
                 >;

using integral_types =
  c2h::type_list<uint16_t,
                 int32_t,
                 uint64_t,
#if _CCCL_HAS_INT128()
                 __uint128_t,
                 __int128_t
#endif // _CCCL_HAS_INT128()
                 >;

/// Sum of integers 1..n (inclusive), as @p T — for PSTL tests with `thrust::sequence(..., 1)`.
template <class T>
TEST_FUNC constexpr T triangular_number(int n)
{
  return static_cast<T>(n * (n + 1) / 2);
}

/// Types safe for PSTL numeric tests at size 1000 (wider than @ref arithmetic_types to avoid overflow in sums).
using pstl_numerics_types =
  c2h::type_list<int32_t,
                 uint64_t,
#if _CCCL_HAS_INT128()
                 __uint128_t,
                 __int128_t,
#endif // _CCCL_HAS_INT128()
                 float,
                 double,
                 nontrivial_type>;

// We want to use thrust::sequence

#if _LIBCUDACXX_HAS_NVFP16()
TEST_FUNC __half operator*(const __half lhs, const size_t rhs) noexcept
{
  return ::__float2half(::__half2float(lhs) * rhs);
}
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
TEST_FUNC __nv_bfloat16 operator*(const __nv_bfloat16 lhs, const size_t rhs) noexcept
{
  return ::__float2bfloat16(::__bfloat162float(lhs) * rhs);
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#endif // SUPPORT_TEST_PSTL_H
