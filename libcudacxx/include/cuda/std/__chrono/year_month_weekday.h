// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_YEAR_MONTH_WEEKDAY_H
#define _CUDA_STD___CHRONO_YEAR_MONTH_WEEKDAY_H

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
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/month.h>
#include <cuda/std/__chrono/month_weekday.h>
#include <cuda/std/__chrono/system_clock.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/__chrono/weekday.h>
#include <cuda/std/__chrono/year.h>
#include <cuda/std/__chrono/year_month.h>
#include <cuda/std/__chrono/year_month_day.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class year_month_weekday
{
  chrono::year __year_;
  chrono::month __month_;
  chrono::weekday_indexed __weekday_indexed_;

  [[nodiscard]] _CCCL_API static constexpr year_month_weekday __from_days(days __days) noexcept
  {
    const sys_days __sysd{__days};
    const chrono::weekday __wd = chrono::weekday{__sysd};
    const year_month_day __ymd = year_month_day{__sysd};
    return year_month_weekday{__ymd.year(), __ymd.month(), __wd[(static_cast<unsigned>(__ymd.day()) - 1) / 7 + 1]};
  }

  [[nodiscard]] _CCCL_API constexpr days __to_days() const noexcept
  {
    const sys_days __sysd{__year_ / __month_ / 1};
    return (__sysd
            + (__weekday_indexed_.weekday() - chrono::weekday(__sysd) + days{(__weekday_indexed_.index() - 1) * 7}))
      .time_since_epoch();
  }

public:
  _CCCL_HIDE_FROM_ABI year_month_weekday() = default;

  _CCCL_API constexpr year_month_weekday(
    const chrono::year& __year, const chrono::month& __month, const chrono::weekday_indexed& __weekday_indexed) noexcept
      : __year_{__year}
      , __month_{__month}
      , __weekday_indexed_{__weekday_indexed}
  {}

  _CCCL_API constexpr year_month_weekday(const sys_days& __sysd) noexcept
      : year_month_weekday(__from_days(__sysd.time_since_epoch()))
  {}

  _CCCL_API explicit constexpr year_month_weekday(const local_days& __locd) noexcept
      : year_month_weekday(__from_days(__locd.time_since_epoch()))
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::year year() const noexcept
  {
    return __year_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::weekday weekday() const noexcept
  {
    return __weekday_indexed_.weekday();
  }

  [[nodiscard]] _CCCL_API constexpr unsigned index() const noexcept
  {
    return __weekday_indexed_.index();
  }

  [[nodiscard]] _CCCL_API constexpr chrono::weekday_indexed weekday_indexed() const noexcept
  {
    return __weekday_indexed_;
  }

  [[nodiscard]] _CCCL_API constexpr operator sys_days() const noexcept
  {
    return sys_days{__to_days()};
  }

  [[nodiscard]] _CCCL_API explicit constexpr operator local_days() const noexcept
  {
    return local_days{__to_days()};
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    if (!__year_.ok() || !__month_.ok() || !__weekday_indexed_.ok())
    {
      return false;
    }
    if (__weekday_indexed_.index() <= 4)
    {
      return true;
    }
    const auto __nth_weekday_day =
      __weekday_indexed_.weekday() - chrono::weekday{static_cast<sys_days>(__year_ / __month_ / 1)}
      + days{(__weekday_indexed_.index() - 1) * 7 + 1};
    return static_cast<unsigned>(__nth_weekday_day.count()) <= static_cast<unsigned>((__year_ / __month_ / last).day());
  }

  // arithmetics
  _CCCL_API constexpr year_month_weekday& operator+=(const months& m) noexcept;
  _CCCL_API constexpr year_month_weekday& operator-=(const months& m) noexcept;
  _CCCL_API constexpr year_month_weekday& operator+=(const years& y) noexcept;
  _CCCL_API constexpr year_month_weekday& operator-=(const years& y) noexcept;

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const year_month_weekday& __lhs, const year_month_weekday& __rhs) noexcept
  {
    return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month()
        && __lhs.weekday_indexed() == __rhs.weekday_indexed();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const year_month_weekday& __lhs, const year_month_weekday& __rhs) noexcept
  {
    return __lhs.year() != __rhs.year() || __lhs.month() != __rhs.month()
        || __lhs.weekday_indexed() != __rhs.weekday_indexed();
  }
#endif // _CCCL_STD_VER <= 2017
};

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator/(const year_month& __lhs, const weekday_indexed& __rhs) noexcept
{
  return year_month_weekday{__lhs.year(), __lhs.month(), __rhs};
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday operator/(const year& __lhs, const month_weekday& __rhs) noexcept
{
  return year_month_weekday{__lhs, __rhs.month(), __rhs.weekday_indexed()};
}

_CCCL_API constexpr year_month_weekday operator/(int __lhs, const month_weekday& __rhs) noexcept
{
  return year(__lhs) / __rhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday operator/(const month_weekday& __lhs, const year& __rhs) noexcept
{
  return __rhs / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday operator/(const month_weekday& __lhs, int __rhs) noexcept
{
  return year(__rhs) / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator+(const year_month_weekday& __lhs, const months& __rhs) noexcept
{
  return (__lhs.year() / __lhs.month() + __rhs) / __lhs.weekday_indexed();
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator+(const months& __lhs, const year_month_weekday& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator-(const year_month_weekday& __lhs, const months& __rhs) noexcept
{
  return __lhs + (-__rhs);
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator+(const year_month_weekday& __lhs, const years& __rhs) noexcept
{
  return year_month_weekday{__lhs.year() + __rhs, __lhs.month(), __lhs.weekday_indexed()};
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator+(const years& __lhs, const year_month_weekday& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday
operator-(const year_month_weekday& __lhs, const years& __rhs) noexcept
{
  return __lhs + (-__rhs);
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday& year_month_weekday::operator+=(const months& __dm) noexcept
{
  *this = *this + __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday& year_month_weekday::operator-=(const months& __dm) noexcept
{
  *this = *this - __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday& year_month_weekday::operator+=(const years& __dy) noexcept
{
  *this = *this + __dy;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday& year_month_weekday::operator-=(const years& __dy) noexcept
{
  *this = *this - __dy;
  return *this;
}

class year_month_weekday_last
{
private:
  chrono::year __year_;
  chrono::month __month_;
  chrono::weekday_last __weekday_last_;

  [[nodiscard]] _CCCL_API constexpr days __to_days() const noexcept
  {
    const sys_days __last{__year_ / __month_ / chrono::last};
    return (__last - (chrono::weekday{__last} - __weekday_last_.weekday())).time_since_epoch();
  }

public:
  _CCCL_API constexpr year_month_weekday_last(
    const chrono::year& __year, const chrono::month& __month, const chrono::weekday_last& __weekday_last) noexcept
      : __year_{__year}
      , __month_{__month}
      , __weekday_last_{__weekday_last}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::year year() const noexcept
  {
    return __year_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::weekday weekday() const noexcept
  {
    return __weekday_last_.weekday();
  }

  [[nodiscard]] _CCCL_API constexpr chrono::weekday_last weekday_last() const noexcept
  {
    return __weekday_last_;
  }

  [[nodiscard]] _CCCL_API constexpr operator sys_days() const noexcept
  {
    return sys_days{__to_days()};
  }

  [[nodiscard]] _CCCL_API explicit constexpr operator local_days() const noexcept
  {
    return local_days{__to_days()};
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __year_.ok() && __month_.ok() && __weekday_last_.ok();
  }

  // arithmetics
  _CCCL_API constexpr year_month_weekday_last& operator+=(const months& __dm) noexcept;
  _CCCL_API constexpr year_month_weekday_last& operator-=(const months& __dm) noexcept;
  _CCCL_API constexpr year_month_weekday_last& operator+=(const years& __dy) noexcept;
  _CCCL_API constexpr year_month_weekday_last& operator-=(const years& __dy) noexcept;

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const year_month_weekday_last& __lhs, const year_month_weekday_last& __rhs) noexcept
  {
    return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month()
        && __lhs.weekday_last() == __rhs.weekday_last();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const year_month_weekday_last& __lhs, const year_month_weekday_last& __rhs) noexcept
  {
    return __lhs.year() != __rhs.year() || __lhs.month() != __rhs.month()
        || __lhs.weekday_last() != __rhs.weekday_last();
  }
#endif // _CCCL_STD_VER <= 2017
};

// Cannot be hidden friends, the compiler fails to find the right operator/
[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator/(const year_month& __lhs, const weekday_last& __rhs) noexcept
{
  return year_month_weekday_last{__lhs.year(), __lhs.month(), __rhs};
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator/(const year& __lhs, const month_weekday_last& __rhs) noexcept
{
  return year_month_weekday_last{__lhs, __rhs.month(), __rhs.weekday_last()};
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last operator/(int __lhs, const month_weekday_last& __rhs) noexcept
{
  return year(__lhs) / __rhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator/(const month_weekday_last& __lhs, const year& __rhs) noexcept
{
  return __rhs / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last operator/(const month_weekday_last& __lhs, int __rhs) noexcept
{
  return year(__rhs) / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator+(const year_month_weekday_last& __lhs, const months& __rhs) noexcept
{
  return (__lhs.year() / __lhs.month() + __rhs) / __lhs.weekday_last();
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator+(const months& __lhs, const year_month_weekday_last& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator-(const year_month_weekday_last& __lhs, const months& __rhs) noexcept
{
  return __lhs + (-__rhs);
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator+(const year_month_weekday_last& __lhs, const years& __rhs) noexcept
{
  return year_month_weekday_last{__lhs.year() + __rhs, __lhs.month(), __lhs.weekday_last()};
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator+(const years& __lhs, const year_month_weekday_last& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last
operator-(const year_month_weekday_last& __lhs, const years& __rhs) noexcept
{
  return __lhs + (-__rhs);
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last&
year_month_weekday_last::operator+=(const months& __dm) noexcept
{
  *this = *this + __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last&
year_month_weekday_last::operator-=(const months& __dm) noexcept
{
  *this = *this - __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last&
year_month_weekday_last::operator+=(const years& __dy) noexcept
{
  *this = *this + __dy;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_weekday_last&
year_month_weekday_last::operator-=(const years& __dy) noexcept
{
  *this = *this - __dy;
  return *this;
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_YEAR_MONTH_WEEKDAY_H
