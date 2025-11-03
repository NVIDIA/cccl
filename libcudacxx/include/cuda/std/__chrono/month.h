// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_MONTH_H
#define _CUDA_STD___CHRONO_MONTH_H

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
class month
{
private:
  unsigned char __m_;

  _CCCL_API constexpr void __add_clamped(long long __add) noexcept
  {
    const auto __mu = static_cast<long long>(static_cast<unsigned>(__m_)) + (__add - 1);
    const auto __yr = (__mu >= 0 ? __mu : __mu - 11) / 12;
    __m_            = static_cast<unsigned char>(__mu - __yr * 12 + 1);
  }

public:
  _CCCL_HIDE_FROM_ABI month() = default;
  _CCCL_API explicit constexpr month(unsigned __val) noexcept
      : __m_(static_cast<unsigned char>(__val))
  {}
  _CCCL_API constexpr month& operator++() noexcept
  {
    __add_clamped(1);
    return *this;
  }
  _CCCL_API constexpr month operator++(int) noexcept
  {
    month __tmp = *this;
    __add_clamped(1);
    return __tmp;
  }
  _CCCL_API constexpr month& operator--() noexcept
  {
    __add_clamped(-1);
    return *this;
  }
  _CCCL_API constexpr month operator--(int) noexcept
  {
    month __tmp = *this;
    __add_clamped(-1);
    return __tmp;
  }

  _CCCL_API constexpr month& operator+=(const months& __dm) noexcept
  {
    __add_clamped(__dm.count());
    return *this;
  }

  _CCCL_API constexpr month& operator-=(const months& __dm) noexcept
  {
    __add_clamped(-__dm.count());
    return *this;
  }

  _CCCL_API explicit constexpr operator unsigned() const noexcept
  {
    return __m_;
  }

  [[nodiscard]] _CCCL_API constexpr bool ok() const noexcept
  {
    return __m_ >= 1 && __m_ <= 12;
  }

  // Arithmetics

  [[nodiscard]] _CCCL_API friend constexpr month operator+(const month& __lhs, const months& __rhs) noexcept
  {
    month __ret{__lhs};
    __ret.__add_clamped(__rhs.count());
    return __ret;
  }

  [[nodiscard]] _CCCL_API friend constexpr month operator+(const months& __lhs, const month& __rhs) noexcept
  {
    month __ret{__rhs};
    __ret.__add_clamped(__lhs.count());
    return __ret;
  }

  [[nodiscard]] _CCCL_API friend constexpr month operator-(const month& __lhs, const months& __rhs) noexcept
  {
    month __ret{__lhs};
    __ret.__add_clamped(-__rhs.count());
    return __ret;
  }

  [[nodiscard]] _CCCL_API friend constexpr months operator-(const month& __lhs, const month& __rhs) noexcept
  {
    auto const __dm = static_cast<unsigned>(__lhs) - static_cast<unsigned>(__rhs);
    return months(__dm <= 11 ? __dm : __dm + 12);
  }

  // Comparisons
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) == static_cast<unsigned>(__rhs);
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) != static_cast<unsigned>(__rhs);
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering operator<=>(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) <=> static_cast<unsigned>(__rhs);
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) < static_cast<unsigned>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) > static_cast<unsigned>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) <= static_cast<unsigned>(__rhs);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const month& __lhs, const month& __rhs) noexcept
  {
    return static_cast<unsigned>(__lhs) >= static_cast<unsigned>(__rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

inline constexpr month January{1};
inline constexpr month February{2};
inline constexpr month March{3};
inline constexpr month April{4};
inline constexpr month May{5};
inline constexpr month June{6};
inline constexpr month July{7};
inline constexpr month August{8};
inline constexpr month September{9};
inline constexpr month October{10};
inline constexpr month November{11};
inline constexpr month December{12};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_MONTH_H
