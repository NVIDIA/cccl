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

#ifndef _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_EXT_H
#define _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_EXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cmath/fp_utils.h>
#include <cuda/std/__limits/numeric_limits.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// nvfp16

#if _CCCL_HAS_NVFP16()
template <>
class __numeric_limits_impl<__half, __numeric_limits_type::__floating_point>
{
public:
  using type = __half;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 11;
  static constexpr int digits10     = 3;
  static constexpr int max_digits10 = 5;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x0400u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x7bffu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0xfbffu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x1400u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x3800u);
  }

  static constexpr int min_exponent   = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent   = 16;
  static constexpr int max_exponent10 = 4;

  static constexpr bool has_infinity             = true;
  static constexpr bool has_quiet_NaN            = true;
  static constexpr bool has_signaling_NaN        = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x7c00u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x7e00u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x7d00u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp16_from_storage(0x0001u);
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP16

// nvbf16

#if _CCCL_HAS_NVBF16()
template <>
class __numeric_limits_impl<__nv_bfloat16, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_bfloat16;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 8;
  static constexpr int digits10     = 2;
  static constexpr int max_digits10 = 4;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x0080u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x7f7fu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0xff7fu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x3c00u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x3f00u);
  }

  static constexpr int min_exponent   = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent   = 128;
  static constexpr int max_exponent10 = 38;

  static constexpr bool has_infinity             = true;
  static constexpr bool has_quiet_NaN            = true;
  static constexpr bool has_signaling_NaN        = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x7f80u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x7fc0u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x7fa0u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvbf16_from_storage(0x0001u);
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVBF16

// nvfp8_e4m3

#if _CCCL_HAS_NVFP8_E4M3()
template <>
class __numeric_limits_impl<__nv_fp8_e4m3, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_fp8_e4m3;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 4;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 3;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0x08u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0x7eu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0xfeu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0x20u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0x30u);
  }

  static constexpr int min_exponent   = -6;
  static constexpr int min_exponent10 = -2;
  static constexpr int max_exponent   = 8;
  static constexpr int max_exponent10 = 2;

  static constexpr bool has_infinity             = false;
  static constexpr bool has_quiet_NaN            = true;
  static constexpr bool has_signaling_NaN        = false;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0x7fu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e4m3_from_storage(0x01u);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP8_E4M3

// nvfp8_e5m2

#if _CCCL_HAS_NVFP8_E5M2()
template <>
class __numeric_limits_impl<__nv_fp8_e5m2, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_fp8_e5m2;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 3;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x04u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x7bu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0xfbu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x34u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x38u);
  }

  static constexpr int min_exponent   = -15;
  static constexpr int min_exponent10 = -5;
  static constexpr int max_exponent   = 15;
  static constexpr int max_exponent10 = 4;

  static constexpr bool has_infinity             = true;
  static constexpr bool has_quiet_NaN            = true;
  static constexpr bool has_signaling_NaN        = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x7cu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x7eu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x7du);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e5m2_from_storage(0x01u);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP8_E5M2

// nvfp8_e8m0

#if _CCCL_HAS_NVFP8_E8M0()
template <>
class __numeric_limits_impl<__nv_fp8_e8m0, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_fp8_e8m0;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = false;
  static constexpr int digits       = 1;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e8m0_from_storage(0x00u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e8m0_from_storage(0xfeu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e8m0_from_storage(0x00u);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e8m0_from_storage(0x7fu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e8m0_from_storage(0x7fu);
  }

  static constexpr int min_exponent   = -127;
  static constexpr int min_exponent10 = -39;
  static constexpr int max_exponent   = 127;
  static constexpr int max_exponent10 = 38;

  static constexpr bool has_infinity             = false;
  static constexpr bool has_quiet_NaN            = true;
  static constexpr bool has_signaling_NaN        = false;
  static constexpr float_denorm_style has_denorm = denorm_absent;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp8_e8m0_from_storage(0xffu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return type{};
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};
#endif // _CCCL_HAS_NVFP8_E8M0()

// nvfp6_e2m3

#if _CCCL_HAS_NVFP6_E2M3()
template <>
class __numeric_limits_impl<__nv_fp6_e2m3, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_fp6_e2m3;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 4;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 3;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e2m3_from_storage(0x08u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e2m3_from_storage(0x1fu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e2m3_from_storage(0x3fu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e2m3_from_storage(0x01u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e2m3_from_storage(0x04u);
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 2;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity             = false;
  static constexpr bool has_quiet_NaN            = false;
  static constexpr bool has_signaling_NaN        = false;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e2m3_from_storage(0x01u);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP6_E2M3

// nvfp6_e3m2

#if _CCCL_HAS_NVFP6_E3M2()
template <>
class __numeric_limits_impl<__nv_fp6_e3m2, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_fp6_e3m2;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 3;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e3m2_from_storage(0x04u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e3m2_from_storage(0x1fu);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e3m2_from_storage(0x3fu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e3m2_from_storage(0x04u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e3m2_from_storage(0x08u);
  }

  static constexpr int min_exponent   = -2;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent   = 4;
  static constexpr int max_exponent10 = 1;

  static constexpr bool has_infinity             = false;
  static constexpr bool has_quiet_NaN            = false;
  static constexpr bool has_signaling_NaN        = false;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp6_e3m2_from_storage(0x01u);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP6_E3M2()

// nvfp4_e2m1

#if _CCCL_HAS_NVFP4_E2M1()
template <>
class __numeric_limits_impl<__nv_fp4_e2m1, __numeric_limits_type::__floating_point>
{
public:
  using type = __nv_fp4_e2m1;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 2;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp4_e2m1_from_storage(0x2u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp4_e2m1_from_storage(0x7u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp4_e2m1_from_storage(0xfu);
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp4_e2m1_from_storage(0x1u);
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp4_e2m1_from_storage(0x1u);
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 2;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity             = false;
  static constexpr bool has_quiet_NaN            = false;
  static constexpr bool has_signaling_NaN        = false;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss          = false;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type infinity() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return _CUDA_VSTD::__cccl_make_nvfp4_e2m1_from_storage(0x1u);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP4_E2M1()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_EXT_H
