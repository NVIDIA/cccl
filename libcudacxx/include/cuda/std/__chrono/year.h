// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_YEAR_H
#define _CUDA_STD___CHRONO_YEAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class year
{
private:
  short __y_;

public:
  _CCCL_HIDE_FROM_ABI year() = default;
  _CCCL_API explicit constexpr year(int __val) noexcept
      : __y_(static_cast<short>(__val))
  {}

  _CCCL_API constexpr year& operator++() noexcept
  {
    ++__y_;
    return *this;
  }
  _CCCL_API constexpr year operator++(int) noexcept
  {
    year __tmp = *this;
    ++__y_;
    return __tmp;
  }
  _CCCL_API constexpr year& operator--() noexcept
  {
    --__y_;
    return *this;
  }
  _CCCL_API constexpr year operator--(int) noexcept
  {
    year __tmp = *this;
    --__y_;
    return __tmp;
  }

  _CCCL_API constexpr year& operator+=(const years& __dy) noexcept
  {
    __y_ = static_cast<short>(static_cast<int>(__y_) + __dy.count());
    return *this;
  }

  _CCCL_API constexpr year& operator-=(const years& __dy) noexcept
  {
    __y_ = static_cast<short>(static_cast<int>(__y_) - __dy.count());
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr year operator+() const noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr year operator-() const noexcept
  {
    return year{-__y_};
  }

  [[nodiscard]] _CCCL_API constexpr bool is_leap() const noexcept
  {
    return __y_ % 4 == 0 && (__y_ % 100 != 0 || __y_ % 400 == 0);
  }

  _CCCL_API explicit constexpr operator int() const noexcept
  {
    return __y_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return static_cast<int>(min()) <= __y_ && __y_ <= static_cast<int>(max());
  }

  [[nodiscard]] _CCCL_API static constexpr year min() noexcept
  {
    return year{-32767};
  }

  [[nodiscard]] _CCCL_API static constexpr year max() noexcept
  {
    return year{32767};
  }

  // Arithmetics
  [[nodiscard]] _CCCL_API friend constexpr year operator+(const year& __lhs, const years& __rhs) noexcept
  {
    return year{static_cast<int>(__lhs) + __rhs.count()};
  }

  [[nodiscard]] _CCCL_API friend constexpr year operator+(const years& __lhs, const year& __rhs) noexcept
  {
    return year{static_cast<int>(__rhs) + __lhs.count()};
  }

  [[nodiscard]] _CCCL_API friend constexpr year operator-(const year& __lhs, const years& __rhs) noexcept
  {
    return year{static_cast<int>(__lhs) - __rhs.count()};
  }

  [[nodiscard]] _CCCL_API friend constexpr years operator-(const year& __lhs, const year& __rhs) noexcept
  {
    return years{static_cast<int>(__lhs) - static_cast<int>(__rhs)};
  }

  // Comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) == static_cast<int>(__rhs);
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) != static_cast<int>(__rhs);
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering operator<=>(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) <=> static_cast<int>(__rhs);
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) < static_cast<int>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) > static_cast<int>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) <= static_cast<int>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const year& __lhs, const year& __rhs) noexcept
  {
    return static_cast<int>(__lhs) >= static_cast<int>(__rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_YEAR_H
