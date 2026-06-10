// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_MONTH_DAY_H
#define _CUDA_STD___CHRONO_MONTH_DAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/calendar.h>
#include <cuda/std/__chrono/day.h>
#include <cuda/std/__chrono/month.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class month_day
{
private:
  chrono::month __month_;
  chrono::day __day_;

public:
  _CCCL_HIDE_FROM_ABI month_day() = default;
  _CCCL_API constexpr month_day(const chrono::month& __month, const chrono::day& __day) noexcept
      : __month_{__month}
      , __day_{__day}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::day day() const noexcept
  {
    return __day_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    if (!__month_.ok())
    {
      return false;
    }
    const unsigned __day = static_cast<unsigned>(__day_);
    if (__day < 1 || __day > 31)
    {
      return false;
    }
    if (__day <= 29)
    {
      return true;
    }
    //  Now we've got either 30 or 31
    const unsigned __month = static_cast<unsigned>(__month_);
    if (__month == 2)
    {
      return false;
    }
    if (__month == 4 || __month == 6 || __month == 9 || __month == 11)
    {
      return __day == 30;
    }
    return true;
  }

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const month_day& __lhs, const month_day& __rhs) noexcept
  {
    return __lhs.month() == __rhs.month() && __lhs.day() == __rhs.day();
  }
#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const month_day& __lhs, const month_day& __rhs) noexcept
  {
    return __lhs.month() != __rhs.month() || __lhs.day() != __rhs.day();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr strong_ordering
  operator<=>(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() <=> __rhs.month() != strong_ordering::equal
           ? __lhs.month() <=> __rhs.month()
           : __lhs.day() <=> __rhs.day();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const month_day& __lhs, const month_day& __rhs) noexcept
  {
    return __lhs.month() != __rhs.month() ? __lhs.month() < __rhs.month() : __lhs.day() < __rhs.day();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const month_day& __lhs, const month_day& __rhs) noexcept
  {
    return __rhs < __lhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const month_day& __lhs, const month_day& __rhs) noexcept
  {
    return !(__rhs < __lhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const month_day& __lhs, const month_day& __rhs) noexcept
  {
    return !(__lhs < __rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

// arithmetics
[[nodiscard]] _CCCL_API constexpr month_day operator/(const chrono::month& __lhs, const chrono::day& __rhs) noexcept
{
  return month_day{__lhs, __rhs};
}

[[nodiscard]] _CCCL_API constexpr month_day operator/(const chrono::day& __lhs, const chrono::month& __rhs) noexcept
{
  return month_day{__rhs, __lhs};
}

[[nodiscard]] _CCCL_API constexpr month_day operator/(const chrono::month& __lhs, int __rhs) noexcept
{
  return month_day{__lhs, chrono::day(__rhs)};
}

[[nodiscard]] _CCCL_API constexpr month_day operator/(int __lhs, const chrono::day& __rhs) noexcept
{
  return month_day{chrono::month(__lhs), __rhs};
}

[[nodiscard]] _CCCL_API constexpr month_day operator/(const chrono::day& __lhs, int __rhs) noexcept
{
  return month_day{chrono::month(__rhs), __lhs};
}

class month_day_last
{
private:
  chrono::month __month_;

public:
  _CCCL_API explicit constexpr month_day_last(const chrono::month& __month) noexcept
      : __month_{__month}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __month_.ok();
  }

  // comparisons

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() == __rhs.month();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() != __rhs.month();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr strong_ordering
  operator<=>(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() <=> __rhs.month();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator<(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() < __rhs.month();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() > __rhs.month();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() <= __rhs.month();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const month_day_last& __lhs, const month_day_last& __rhs) noexcept
  {
    return __lhs.month() >= __rhs.month();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

// arithmetics
[[nodiscard]] _CCCL_API constexpr month_day_last operator/(const chrono::month& __lhs, last_spec) noexcept
{
  return month_day_last{__lhs};
}

[[nodiscard]] _CCCL_API constexpr month_day_last operator/(last_spec, const chrono::month& __rhs) noexcept
{
  return month_day_last{__rhs};
}

[[nodiscard]] _CCCL_API constexpr month_day_last operator/(int __lhs, last_spec) noexcept
{
  return month_day_last{chrono::month(__lhs)};
}

[[nodiscard]] _CCCL_API constexpr month_day_last operator/(last_spec, int __rhs) noexcept
{
  return month_day_last{chrono::month(__rhs)};
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_MONTH_DAY_H
