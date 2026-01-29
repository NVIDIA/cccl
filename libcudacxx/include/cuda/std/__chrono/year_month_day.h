// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_YEAR_MONTH_DAY_H
#define _CUDA_STD___CHRONO_YEAR_MONTH_DAY_H

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
#include <cuda/std/__chrono/month_day.h>
#include <cuda/std/__chrono/system_clock.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/__chrono/year.h>
#include <cuda/std/__chrono/year_month.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class year_month_day_last;

class year_month_day
{
private:
  chrono::year __year_;
  chrono::month __month_;
  chrono::day __day_;

  // https://howardhinnant.github.io/date_algorithms.html#days_from_civil
  [[nodiscard]] _CCCL_API constexpr days __to_days() const noexcept
  {
    static_assert(cuda::std::numeric_limits<unsigned>::digits >= 18, "");
    static_assert(cuda::std::numeric_limits<int>::digits >= 20, "");

    // nvcc doesn't allow ODR using constexpr globals. Therefore,
    // make a temporary initialized from the global
    auto constexpr __Feb = February;
    const int __yr       = static_cast<int>(__year_) - (__month_ <= __Feb);
    const unsigned __mth = static_cast<unsigned>(__month_);
    const unsigned __day = static_cast<unsigned>(__day_);

    const int __era      = (__yr >= 0 ? __yr : __yr - 399) / 400;
    const unsigned __yoe = static_cast<unsigned>(__yr - __era * 400); // [0, 399]
    const unsigned __doy =
      static_cast<unsigned>((153 * (__mth + static_cast<unsigned>(__mth > 2 ? -3 : 9)) + 2) / 5 + __day - 1); // [0,
                                                                                                              // 365]
    const unsigned __doe = __yoe * 365 + __yoe / 4 - __yoe / 100 + __doy; // [0, 146096]
    return days{__era * 146097 + static_cast<int>(__doe) - 719468};
  }

  // https://howardhinnant.github.io/date_algorithms.html#civil_from_days
  [[nodiscard]] _CCCL_API static constexpr year_month_day __from_days(days __d) noexcept
  {
    static_assert(cuda::std::numeric_limits<unsigned>::digits >= 18, "");
    static_assert(cuda::std::numeric_limits<int>::digits >= 20, "");
    const int __z        = __d.count() + 719468;
    const int __era      = (__z >= 0 ? __z : __z - 146096) / 146097;
    const unsigned __doe = static_cast<unsigned>(__z - __era * 146097); // [0, 146096]
    const unsigned __yoe = (__doe - __doe / 1460 + __doe / 36524 - __doe / 146096) / 365; // [0, 399]
    const int __yr       = static_cast<int>(__yoe) + __era * 400;
    const unsigned __doy = __doe - (365 * __yoe + __yoe / 4 - __yoe / 100); // [0, 365]
    const unsigned __mp  = (5 * __doy + 2) / 153; // [0, 11]
    const unsigned __day = __doy - (153 * __mp + 2) / 5 + 1; // [1, 31]
    const unsigned __mth = __mp + static_cast<unsigned>(__mp < 10 ? 3 : -9); // [1, 12]
    return year_month_day{chrono::year{__yr + (__mth <= 2)}, chrono::month{__mth}, chrono::day{__day}};
  }

public:
  _CCCL_HIDE_FROM_ABI year_month_day() = default;

  _CCCL_API constexpr year_month_day(
    const chrono::year& __year, const chrono::month& __month, const chrono::day& __day) noexcept
      : __year_{__year}
      , __month_{__month}
      , __day_{__day}
  {}

  _CCCL_API constexpr year_month_day(const year_month_day_last& __ymdl) noexcept;

  _CCCL_API constexpr year_month_day(const sys_days& __sysd) noexcept
      : year_month_day(__from_days(__sysd.time_since_epoch()))
  {}

  _CCCL_API explicit constexpr year_month_day(const local_days& __locd) noexcept
      : year_month_day(__from_days(__locd.time_since_epoch()))
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::year year() const noexcept
  {
    return __year_;
  }
  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }
  [[nodiscard]] _CCCL_API constexpr chrono::day day() const noexcept
  {
    return __day_;
  }
  [[nodiscard]] _CCCL_API constexpr operator sys_days() const noexcept
  {
    return sys_days{__to_days()};
  }
  [[nodiscard]] _CCCL_API explicit constexpr operator local_days() const noexcept
  {
    return local_days{__to_days()};
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept;

  // arithmetics
  _CCCL_API constexpr year_month_day& operator+=(const months& __dm) noexcept;
  _CCCL_API constexpr year_month_day& operator-=(const months& __dm) noexcept;
  _CCCL_API constexpr year_month_day& operator+=(const years& __dy) noexcept;
  _CCCL_API constexpr year_month_day& operator-=(const years& __dy) noexcept;

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    return __lhs.year() == __rhs.year() && __lhs.month() == __rhs.month() && __lhs.day() == __rhs.day();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    return __lhs.year() != __rhs.year() || __lhs.month() != __rhs.month() || __lhs.day() != __rhs.day();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr strong_ordering
  operator<=>(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    if (__lhs.year() <=> __rhs.year() != strong_ordering::equal)
    {
      return __lhs.year() <=> __rhs.year();
    }
    else if (__lhs.month() <=> __rhs.month() != strong_ordering::equal)
    {
      return __lhs.month() <=> __rhs.month()
    }
    else
    {
      __lhs.day() <=> __rhs.day()
    }
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  [[nodiscard]]
  _CCCL_API friend constexpr bool operator<(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    if (__lhs.year() < __rhs.year())
    {
      return true;
    }
    if (__lhs.year() > __rhs.year())
    {
      return false;
    }
    if (__lhs.month() < __rhs.month())
    {
      return true;
    }
    if (__lhs.month() > __rhs.month())
    {
      return false;
    }
    return __lhs.day() < __rhs.day();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    return __rhs < __lhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    return !(__rhs < __lhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const year_month_day& __lhs, const year_month_day& __rhs) noexcept
  {
    return !(__lhs < __rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

// Cannot be hidden friends, the compiler fails to find the right operator/
[[nodiscard]] _CCCL_API constexpr year_month_day operator/(const year_month& __lhs, const day& __rhs) noexcept
{
  return year_month_day{__lhs.year(), __lhs.month(), __rhs};
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator/(const year_month& __lhs, int __rhs) noexcept
{
  return year_month_day{__lhs.year(), __lhs.month(), chrono::day(__rhs)};
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator/(const year& __lhs, const month_day& __rhs) noexcept
{
  return __lhs / __rhs.month() / __rhs.day();
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator/(int __lhs, const month_day& __rhs) noexcept
{
  return year(__lhs) / __rhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator/(const month_day& __lhs, const year& __rhs) noexcept
{
  return __rhs / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator/(const month_day& __lhs, int __rhs) noexcept
{
  return year(__rhs) / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator+(const year_month_day& __lhs, const months& __rhs) noexcept
{
  return (__lhs.year() / __lhs.month() + __rhs) / __lhs.day();
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator+(const months& __lhs, const year_month_day& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator-(const year_month_day& __lhs, const months& __rhs) noexcept
{
  return __lhs + -__rhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator+(const year_month_day& __lhs, const years& __rhs) noexcept
{
  return (__lhs.year() + __rhs) / __lhs.month() / __lhs.day();
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator+(const years& __lhs, const year_month_day& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day operator-(const year_month_day& __lhs, const years& __rhs) noexcept
{
  return __lhs + -__rhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day& year_month_day::operator+=(const months& __dm) noexcept
{
  *this = *this + __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_day& year_month_day::operator-=(const months& __dm) noexcept
{
  *this = *this - __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_day& year_month_day::operator+=(const years& __dy) noexcept
{
  *this = *this + __dy;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_day& year_month_day::operator-=(const years& __dy) noexcept
{
  *this = *this - __dy;
  return *this;
}

class year_month_day_last
{
private:
  chrono::year __year_;
  chrono::month_day_last __month_day_last_;

public:
  _CCCL_API constexpr year_month_day_last(const chrono::year& __year,
                                          const chrono::month_day_last& __month_day_last) noexcept
      : __year_{__year}
      , __month_day_last_{__month_day_last}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::year year() const noexcept
  {
    return __year_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_day_last_.month();
  }

  [[nodiscard]] _CCCL_API constexpr chrono::month_day_last month_day_last() const noexcept
  {
    return __month_day_last_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::day day() const noexcept
  {
    constexpr chrono::day __days_per_month_[] = {
      chrono::day{31},
      chrono::day{28},
      chrono::day{31},
      chrono::day{30},
      chrono::day{31},
      chrono::day{30},
      chrono::day{31},
      chrono::day{31},
      chrono::day{30},
      chrono::day{31},
      chrono::day{30},
      chrono::day{31}};

    // nvcc doesn't allow ODR using constexpr globals. Therefore,
    // make a temporary initialized from the global
    auto constexpr __Feb = February;
    return month() != __Feb || !__year_.is_leap()
           ? __days_per_month_[static_cast<unsigned>(month()) - 1]
           : chrono::day{29};
  }

  [[nodiscard]] _CCCL_API constexpr operator sys_days() const noexcept
  {
    return sys_days{year() / month() / day()};
  }

  [[nodiscard]] _CCCL_API explicit constexpr operator local_days() const noexcept
  {
    return local_days{year() / month() / day()};
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __year_.ok() && __month_day_last_.ok();
  }

  // arithmetics
  _CCCL_API constexpr year_month_day_last& operator+=(const months& __month_) noexcept;
  _CCCL_API constexpr year_month_day_last& operator-=(const months& __month_) noexcept;
  _CCCL_API constexpr year_month_day_last& operator+=(const years& __year_) noexcept;
  _CCCL_API constexpr year_month_day_last& operator-=(const years& __year_) noexcept;

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    return __lhs.year() == __rhs.year() && __lhs.month_day_last() == __rhs.month_day_last();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    return __lhs.year() != __rhs.year() || __lhs.month_day_last() != __rhs.month_day_last();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr strong_ordering
  operator<=>(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    return (__lhs.year() <=> __rhs.year() != strong_ordering::equal)
           ? __lhs.year() <=> __rhs.year()
           : __lhs.month_day_last() <=> __rhs.month_day_last();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator<(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    if (__lhs.year() < __rhs.year())
    {
      return true;
    }
    if (__lhs.year() > __rhs.year())
    {
      return false;
    }
    return __lhs.month_day_last() < __rhs.month_day_last();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    return __rhs < __lhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    return !(__rhs < __lhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const year_month_day_last& __lhs, const year_month_day_last& __rhs) noexcept
  {
    return !(__lhs < __rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

// Cannot be hidden friends, the compiler fails to find the right operator/
[[nodiscard]] _CCCL_API constexpr year_month_day_last operator/(const year_month& __lhs, last_spec) noexcept
{
  return year_month_day_last{__lhs.year(), month_day_last{__lhs.month()}};
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last operator/(const year& __lhs, const month_day_last& __rhs) noexcept
{
  return year_month_day_last{__lhs, __rhs};
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last operator/(int __lhs, const month_day_last& __rhs) noexcept
{
  return year_month_day_last{year{__lhs}, __rhs};
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last operator/(const month_day_last& __lhs, const year& __rhs) noexcept
{
  return __rhs / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last operator/(const month_day_last& __lhs, int __rhs) noexcept
{
  return year{__rhs} / __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last
operator+(const year_month_day_last& __lhs, const months& __rhs) noexcept
{
  return (__lhs.year() / __lhs.month() + __rhs) / last;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last
operator+(const months& __lhs, const year_month_day_last& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last
operator-(const year_month_day_last& __lhs, const months& __rhs) noexcept
{
  return __lhs + (-__rhs);
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last
operator+(const year_month_day_last& __lhs, const years& __rhs) noexcept
{
  return year_month_day_last{__lhs.year() + __rhs, __lhs.month_day_last()};
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last
operator+(const years& __lhs, const year_month_day_last& __rhs) noexcept
{
  return __rhs + __lhs;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last
operator-(const year_month_day_last& __lhs, const years& __rhs) noexcept
{
  return __lhs + (-__rhs);
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last& year_month_day_last::operator+=(const months& __dm) noexcept
{
  *this = *this + __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last& year_month_day_last::operator-=(const months& __dm) noexcept
{
  *this = *this - __dm;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last& year_month_day_last::operator+=(const years& __dy) noexcept
{
  *this = *this + __dy;
  return *this;
}

[[nodiscard]] _CCCL_API constexpr year_month_day_last& year_month_day_last::operator-=(const years& __dy) noexcept
{
  *this = *this - __dy;
  return *this;
}

_CCCL_API constexpr year_month_day::year_month_day(const year_month_day_last& __ymdl) noexcept
    : __year_{__ymdl.year()}
    , __month_{__ymdl.month()}
    , __day_{__ymdl.day()}
{}

[[nodiscard]] _CCCL_API constexpr bool year_month_day::ok() const noexcept
{
  if (!__year_.ok() || !__month_.ok())
  {
    return false;
  }
  return chrono::day{1} <= __day_ && __day_ <= (__year_ / __month_ / last).day();
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_YEAR_MONTH_DAY_H
