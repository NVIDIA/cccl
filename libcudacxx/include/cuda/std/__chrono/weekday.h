// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_WEEKDAY_H
#define _CUDA_STD___CHRONO_WEEKDAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/calendar.h>
#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/system_clock.h>
#include <cuda/std/__chrono/time_point.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class weekday_indexed;
class weekday_last;

class weekday
{
private:
  unsigned char __weekday_;

  // https://howardhinnant.github.io/date_algorithms.html#weekday_from_days
  [[nodiscard]] _CCCL_API static constexpr unsigned char __weekday_from_days(int __days) noexcept
  {
    return static_cast<unsigned char>(static_cast<unsigned>(__days >= -4 ? (__days + 4) % 7 : (__days + 5) % 7 + 6));
  }

public:
  _CCCL_HIDE_FROM_ABI weekday() = default;
  _CCCL_API explicit constexpr weekday(const unsigned __weekday) noexcept
      : __weekday_(static_cast<unsigned char>(__weekday == 7 ? 0 : __weekday))
  {}
  _CCCL_API constexpr weekday(const sys_days& __sysd) noexcept
      : __weekday_(__weekday_from_days(__sysd.time_since_epoch().count()))
  {}
  _CCCL_API explicit constexpr weekday(const local_days& __locd) noexcept
      : __weekday_(__weekday_from_days(__locd.time_since_epoch().count()))
  {}

  _CCCL_API constexpr weekday& operator++() noexcept
  {
    __weekday_ = (__weekday_ == 6 ? 0 : __weekday_ + 1);
    return *this;
  }
  _CCCL_API constexpr weekday operator++(int) noexcept
  {
    weekday __tmp = *this;
    ++(*this);
    return __tmp;
  }
  _CCCL_API constexpr weekday& operator--() noexcept
  {
    __weekday_ = (__weekday_ == 0 ? 6 : __weekday_ - 1);
    return *this;
  }
  _CCCL_API constexpr weekday operator--(int) noexcept
  {
    weekday __tmp = *this;
    --(*this);
    return __tmp;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned c_encoding() const noexcept
  {
    return __weekday_;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned iso_encoding() const noexcept
  {
    return __weekday_ == 0u ? 7 : __weekday_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __weekday_ <= 6;
  }

  [[nodiscard]] _CCCL_API constexpr weekday_indexed operator[](unsigned __index) const noexcept;
  [[nodiscard]] _CCCL_API constexpr weekday_last operator[](last_spec) const noexcept;

  // Arithmetics

  _CCCL_API constexpr weekday& operator+=(const days& __dd) noexcept
  {
    *this = *this + __dd;
    return *this;
  }

  _CCCL_API constexpr weekday& operator-=(const days& __dd) noexcept
  {
    *this = *this - __dd;
    return *this;
  }

  [[nodiscard]] _CCCL_API friend constexpr days operator-(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    // casts are required to work around nvcc bug 3145483
    const int __wdu = static_cast<int>(__lhs.c_encoding()) - static_cast<int>(__rhs.c_encoding());
    const int __wk  = (__wdu >= 0 ? __wdu : __wdu - 6) / 7;
    return days{__wdu - __wk * 7};
  }

  [[nodiscard]] _CCCL_API friend constexpr weekday operator+(const weekday& __lhs, const days& __rhs) noexcept
  {
    auto const __mu = static_cast<long long>(__lhs.c_encoding()) + __rhs.count();
    auto const __yr = (__mu >= 0 ? __mu : __mu - 6) / 7;
    return weekday{static_cast<unsigned>(__mu - __yr * 7)};
  }

  [[nodiscard]] _CCCL_API friend constexpr weekday operator+(const days& __lhs, const weekday& __rhs) noexcept
  {
    return __rhs + __lhs;
  }

  [[nodiscard]] _CCCL_API friend constexpr weekday operator-(const weekday& __lhs, const days& __rhs) noexcept
  {
    return __lhs + -__rhs;
  }

  // Comparisons

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() == __rhs.c_encoding();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() != __rhs.c_encoding();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr strong_ordering operator<=>(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() <=> __rhs.c_encoding();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() < __rhs.c_encoding();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() > __rhs.c_encoding();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() <= __rhs.c_encoding();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const weekday& __lhs, const weekday& __rhs) noexcept
  {
    return __lhs.c_encoding() >= __rhs.c_encoding();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

class weekday_indexed
{
private:
  chrono::weekday __weekday_;
  unsigned char __idx_;

public:
  _CCCL_HIDE_FROM_ABI weekday_indexed() = default;
  _CCCL_API constexpr weekday_indexed(const chrono::weekday& __weekday, unsigned __idx) noexcept
      : __weekday_{__weekday}
      , __idx_(static_cast<unsigned char>(__idx))
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::weekday weekday() const noexcept
  {
    return __weekday_;
  }

  [[nodiscard]] _CCCL_API constexpr unsigned index() const noexcept
  {
    return __idx_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __weekday_.ok() && __idx_ >= 1 && __idx_ <= 5;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const weekday_indexed& __lhs, const weekday_indexed& __rhs) noexcept
  {
    return __lhs.weekday() == __rhs.weekday() && __lhs.index() == __rhs.index();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const weekday_indexed& __lhs, const weekday_indexed& __rhs) noexcept
  {
    return __lhs.weekday() != __rhs.weekday() || __lhs.index() != __rhs.index();
  }
#endif // _CCCL_STD_VER <= 2017
};

class weekday_last
{
private:
  chrono::weekday __weekday_;

public:
  _CCCL_API explicit constexpr weekday_last(const chrono::weekday& __weekday) noexcept
      : __weekday_{__weekday}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::weekday weekday() const noexcept
  {
    return __weekday_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __weekday_.ok();
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const weekday_last& __lhs, const weekday_last& __rhs) noexcept
  {
    return __lhs.weekday() == __rhs.weekday();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const weekday_last& __lhs, const weekday_last& __rhs) noexcept
  {
    return __lhs.weekday() != __rhs.weekday();
  }
#endif // _CCCL_STD_VER <= 2017
};

[[nodiscard]] _CCCL_API constexpr weekday_indexed weekday::operator[](unsigned __index) const noexcept
{
  return weekday_indexed{*this, __index};
}

[[nodiscard]] _CCCL_API constexpr weekday_last weekday::operator[](last_spec) const noexcept
{
  return weekday_last{*this};
}

inline constexpr weekday Sunday{0};
inline constexpr weekday Monday{1};
inline constexpr weekday Tuesday{2};
inline constexpr weekday Wednesday{3};
inline constexpr weekday Thursday{4};
inline constexpr weekday Friday{5};
inline constexpr weekday Saturday{6};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_WEEKDAY_H
