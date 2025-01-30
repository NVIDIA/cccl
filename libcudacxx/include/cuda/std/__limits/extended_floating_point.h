// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___LIMITS_EXTENDED_FLOATING_POINT_H
#define _LIBCUDACXX___LIMITS_EXTENDED_FLOATING_POINT_H

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__floating_point/fp.h>
#include <cuda/std/__limits/limits.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _FpConfig>
class __numeric_limits_impl<::cuda::__fp<_FpConfig>, __numeric_limits_type::__floating_point>
{
public:
  using type = ::cuda::__fp<_FpConfig>;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = type::__is_signed;
  static constexpr int digits       = type::__mant_nbits;
  static constexpr int digits10     = 0; // todo
  static constexpr int max_digits10 = 2 + (digits * 30103l) / 100000l;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return type::__min();
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return type::__max();
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return type::__lowest();
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return type::__epsilon();
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return type{0.5};
  }

  static constexpr int min_exponent   = -(1 << (type::__exp_nbits - 1)) + 3;
  static constexpr int min_exponent10 = 0; // todo
  static constexpr int max_exponent   = 1 << (type::__exp_nbits - 1);
  static constexpr int max_exponent10 = 0; // todo

  static constexpr bool has_infinity             = type::__has_inf;
  static constexpr bool has_quiet_NaN            = type::__has_nan;
  static constexpr bool has_signaling_NaN        = type::__has_nans;
  static constexpr float_denorm_style has_denorm = (type::__has_denorm) ? denorm_present : denorm_absent;
  static constexpr bool has_denorm_loss          = false;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    if constexpr (has_infinity)
    {
      return type::__inf();
    }
    else
    {
      return type{};
    }
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    if constexpr (has_quiet_NaN)
    {
      return type::__nan();
    }
    else
    {
      return type{};
    }
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    if constexpr (has_signaling_NaN)
    {
      return type::__nans();
    }
    else
    {
      return type{};
    }
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return type::__denorm_min();
  }

  static constexpr bool is_iec559  = _FpConfig::__is_iec559;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LIMITS_EXTENDED_FLOATING_POINT_H
