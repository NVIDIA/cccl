// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_MONTH_WEEKDAY_H
#define _CUDA_STD___CHRONO_MONTH_WEEKDAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/month.h>
#include <cuda/std/__chrono/weekday.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class month_weekday
{
private:
  chrono::month __month_;
  chrono::weekday_indexed __weekday_indexed_;

public:
  _CCCL_HIDE_FROM_ABI month_weekday() = default;
  _CCCL_API constexpr month_weekday(const chrono::month& __month,
                                    const chrono::weekday_indexed& __weekday_indexed) noexcept
      : __month_{__month}
      , __weekday_indexed_{__weekday_indexed}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::weekday_indexed weekday_indexed() const noexcept
  {
    return __weekday_indexed_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __month_.ok() && __weekday_indexed_.ok();
  }

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const month_weekday& __lhs, const month_weekday& __rhs) noexcept
  {
    return __lhs.month() == __rhs.month() && __lhs.weekday_indexed() == __rhs.weekday_indexed();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const month_weekday& __lhs, const month_weekday& __rhs) noexcept
  {
    return __lhs.month() != __rhs.month() || __lhs.weekday_indexed() != __rhs.weekday_indexed();
  }
#endif // _CCCL_STD_VER <= 2017
};

// arithmetics
[[nodiscard]] _CCCL_API constexpr month_weekday
operator/(const chrono::month& __lhs, const chrono::weekday_indexed& __rhs) noexcept
{
  return month_weekday{__lhs, __rhs};
}

[[nodiscard]] _CCCL_API constexpr month_weekday operator/(int __lhs, const chrono::weekday_indexed& __rhs) noexcept
{
  return month_weekday{chrono::month(__lhs), __rhs};
}

[[nodiscard]] _CCCL_API constexpr month_weekday
operator/(const chrono::weekday_indexed& __lhs, const chrono::month& __rhs) noexcept
{
  return month_weekday{__rhs, __lhs};
}

[[nodiscard]] _CCCL_API constexpr month_weekday operator/(const chrono::weekday_indexed& __lhs, int __rhs) noexcept
{
  return month_weekday{chrono::month(__rhs), __lhs};
}

class month_weekday_last
{
  chrono::month __month_;
  chrono::weekday_last __weekday_last_;

public:
  _CCCL_API constexpr month_weekday_last(const chrono::month& __month,
                                         const chrono::weekday_last& __weekday_last) noexcept
      : __month_{__month}
      , __weekday_last_{__weekday_last}
  {}

  [[nodiscard]] _CCCL_API constexpr chrono::month month() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::weekday_last weekday_last() const noexcept
  {
    return __weekday_last_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __month_.ok() && __weekday_last_.ok();
  }

  // comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const month_weekday_last& __lhs, const month_weekday_last& __rhs) noexcept
  {
    return __lhs.month() == __rhs.month() && __lhs.weekday_last() == __rhs.weekday_last();
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const month_weekday_last& __lhs, const month_weekday_last& __rhs) noexcept
  {
    return __lhs.month() != __rhs.month() || __lhs.weekday_last() != __rhs.weekday_last();
  }
#endif // _CCCL_STD_VER <= 2017
};

// arithhmetics
[[nodiscard]] _CCCL_API constexpr month_weekday_last
operator/(const chrono::month& __lhs, const chrono::weekday_last& __rhs) noexcept
{
  return month_weekday_last{__lhs, __rhs};
}

[[nodiscard]] _CCCL_API constexpr month_weekday_last operator/(int __lhs, const chrono::weekday_last& __rhs) noexcept
{
  return month_weekday_last{chrono::month(__lhs), __rhs};
}

[[nodiscard]] _CCCL_API constexpr month_weekday_last
operator/(const chrono::weekday_last& __lhs, const chrono::month& __rhs) noexcept
{
  return month_weekday_last{__rhs, __lhs};
}

[[nodiscard]] _CCCL_API constexpr month_weekday_last operator/(const chrono::weekday_last& __lhs, int __rhs) noexcept
{
  return month_weekday_last{chrono::month(__rhs), __lhs};
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_MONTH_WEEKDAY_H
