// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_YEAR_MONTH_H
#define _CUDA_STD___CHRONO_YEAR_MONTH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/month.h>
#include <cuda/std/__chrono/year.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class year_month
{
  chrono::year __year_;
  chrono::month __month_;

public:
  _CCCL_HIDE_FROM_ABI year_month() = default;
  _CCCL_API constexpr year_month(const chrono::year& __year, const chrono::month& __month) noexcept
      : __year_{__year}
      , __month_{__month}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::year year() const noexcept
  {
    return __year_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __year_.ok() && __month_.ok();
  }

  // arithmetics
  _CCCL_API constexpr year_month& operator+=(const months& __months) noexcept
  {
    __month_ += __months;
    return *this;
  }

  _CCCL_API constexpr year_month& operator-=(const months& __months) noexcept
  {
    __month_ -= __months;
    return *this;
  }

  _CCCL_API constexpr year_month& operator+=(const years& __years) noexcept
  {
    __year_ += __years;
    return *this;
  }
  _CCCL_API constexpr year_month& operator-=(const years& __years) noexcept
  {
    __year_ -= __years;
    return *this;
  }

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return __lhs.year() != __rhs.year() || __lhs.month() != __rhs.month();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr strong_ordering operator<=>(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return __lhs.year() <=> __rhs.year() != strong_ordering::equal
           ? __lhs.year() <=> __rhs.year()
           : __lhs.month() <=> __rhs.month();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return __lhs.year() != __rhs.year() ? __lhs.year() < __rhs.year() : __lhs.month() < __rhs.month();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return __rhs < __lhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return !(__rhs < __lhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const year_month& __lhs, const year_month& __rhs) noexcept
  {
    return !(__lhs < __rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

// Cannot be hidden friends, the compiler fails to find the right operator/
[[nodiscard]] _CCCL_API constexpr year_month operator/(const chrono::year& __year, const chrono::month& __month) noexcept
{
  return year_month{__year, __month};
}

[[nodiscard]] _CCCL_API constexpr year_month operator/(const chrono::year& __year, int __month) noexcept
{
  return year_month{__year, chrono::month(__month)};
}

[[nodiscard]] _CCCL_API constexpr year_month operator+(const year_month& __lhs, const months& __rhs) noexcept
{
  int __dmi         = static_cast<int>(static_cast<unsigned>(__lhs.month())) - 1 + __rhs.count();
  const int __years = (__dmi >= 0 ? __dmi : __dmi - 11) / 12;
  __dmi             = __dmi - __years * 12 + 1;
  return (__lhs.year() + years(__years)) / chrono::month(static_cast<unsigned>(__dmi));
}

[[nodiscard]] _CCCL_API constexpr year_month operator+(const months& __lhs, const year_month& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month operator+(const year_month& __lhs, const years& __rhs) noexcept
{
  return (__lhs.year() + __rhs) / __lhs.month();
}

[[nodiscard]] _CCCL_API constexpr year_month operator+(const years& __lhs, const year_month& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr months operator-(const year_month& __lhs, const year_month& __rhs) noexcept
{
  return (__lhs.year() - __rhs.year())
       + chrono::months(static_cast<unsigned>(__lhs.month()) - static_cast<unsigned>(__rhs.month()));
}

[[nodiscard]] _CCCL_API constexpr year_month operator-(const year_month& __lhs, const months& __rhs) noexcept
{
  return __lhs + -__rhs;
}

[[nodiscard]] _CCCL_API constexpr year_month operator-(const year_month& __lhs, const years& __rhs) noexcept
{
  return __lhs + -__rhs;
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_YEAR_MONTH_H
