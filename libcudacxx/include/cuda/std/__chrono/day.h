// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_DAY_H
#define _CUDA_STD___CHRONO_DAY_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class day
{
private:
  unsigned char __d_;

public:
  _CCCL_HIDE_FROM_ABI day() = default;
  _CCCL_API explicit constexpr day(unsigned __val) noexcept
      : __d_(static_cast<unsigned char>(__val))
  {}

  _CCCL_API constexpr day& operator++() noexcept
  {
    ++__d_;
    return *this;
  }
  _CCCL_API constexpr day operator++(int) noexcept
  {
    day __tmp = *this;
    ++__d_;
    return __tmp;
  }
  _CCCL_API constexpr day& operator--() noexcept
  {
    --__d_;
    return *this;
  }
  _CCCL_API constexpr day operator--(int) noexcept
  {
    day __tmp = *this;
    --__d_;
    return __tmp;
  }

  _CCCL_API constexpr day& operator+=(const days& __dd) noexcept
  {
    __d_ += static_cast<unsigned char>(__dd.count());
    return *this;
  }

  _CCCL_API constexpr day& operator-=(const days& __dd) noexcept
  {
    __d_ -= static_cast<unsigned char>(__dd.count());
    return *this;
  }

  _CCCL_API explicit constexpr operator unsigned() const noexcept
  {
    return __d_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __d_ >= 1 && __d_ <= 31;
  }

  // Arithmetics

  [[nodiscard]] _CCCL_API friend constexpr day operator+(const day& __lhs, const days& __rhs) noexcept
  {
    return day{static_cast<unsigned>(__lhs) + static_cast<unsigned>(__rhs.count())};
  }

  [[nodiscard]] _CCCL_API friend constexpr day operator+(const days& __lhs, const day& __rhs) noexcept
  {
    return day{static_cast<unsigned>(__lhs.count()) + static_cast<unsigned>(__rhs)};
  }

  [[nodiscard]] _CCCL_API friend constexpr day operator-(const day& __lhs, const days& __rhs) noexcept
  {
    return day{static_cast<unsigned>(__lhs) - static_cast<unsigned>(__rhs.count())};
  }

  [[nodiscard]] _CCCL_API friend constexpr days operator-(const day& __lhs, const day& __rhs) noexcept
  {
    return days{static_cast<int>(static_cast<unsigned>(__lhs)) - static_cast<int>(static_cast<unsigned>(__rhs))};
  }

  // Comparisons

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) == static_cast<unsigned>(__rhs);
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) != static_cast<unsigned>(__rhs);
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering operator<=>(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) <=> static_cast<unsigned>(__rhs);
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) < static_cast<unsigned>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) > static_cast<unsigned>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) <= static_cast<unsigned>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const day& __lhs, const day& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) >= static_cast<unsigned>(__rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_DAY_H
