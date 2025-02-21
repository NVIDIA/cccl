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
#include <cuda/std/__limits/numeric_limits.h>

#if defined(_LIBCUDACXX_HAS_NVFP16)
#  include <cuda_fp16.h>
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif // _LIBCUDACXX_HAS_NVBF16

#if _CCCL_HAS_NVFP8()
#  include <cuda_fp8.h>
#endif // _CCCL_HAS_NVFP8()

#if _CCCL_HAS_NVFP6()
#  include <cuda_fp6.h>
#endif // _CCCL_HAS_NVFP6()

#if _CCCL_HAS_NVFP4()
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-parameter")
_CCCL_DIAG_SUPPRESS_MSVC(4100) // unreferenced formal parameter
#  include <cuda_fp4.h>
_CCCL_DIAG_POP
#endif // _CCCL_HAS_NVFP4()

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_NVFP16()
#  ifdef _LIBCUDACXX_HAS_NVFP16
#    define _LIBCUDACXX_FP16_CONSTEXPR constexpr
#  else //_LIBCUDACXX_HAS_NVFP16
#    define _LIBCUDACXX_FP16_CONSTEXPR
#  endif //_LIBCUDACXX_HAS_NVFP16

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
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type min() noexcept
  {
    return type(__half_raw{0x0400u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type max() noexcept
  {
    return type(__half_raw{0x7bffu});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type lowest() noexcept
  {
    return type(__half_raw{0xfbffu});
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type epsilon() noexcept
  {
    return type(__half_raw{0x1400u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type round_error() noexcept
  {
    return type(__half_raw{0x3800u});
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
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type infinity() noexcept
  {
    return type(__half_raw{0x7c00u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type quiet_NaN() noexcept
  {
    return type(__half_raw{0x7e00u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type signaling_NaN() noexcept
  {
    return type(__half_raw{0x7d00u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_FP16_CONSTEXPR type denorm_min() noexcept
  {
    return type(__half_raw{0x0001u});
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#  undef _LIBCUDACXX_FP16_CONSTEXPR
#endif // _CCCL_HAS_NVFP16

#if _CCCL_HAS_NVBF16()
#  ifdef _LIBCUDACXX_HAS_NVBF16
#    define _LIBCUDACXX_BF16_CONSTEXPR constexpr
#  else //_LIBCUDACXX_HAS_NVBF16
#    define _LIBCUDACXX_BF16_CONSTEXPR
#  endif //_LIBCUDACXX_HAS_NVBF16

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
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type min() noexcept
  {
    return type(__nv_bfloat16_raw{0x0080u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type max() noexcept
  {
    return type(__nv_bfloat16_raw{0x7f7fu});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type lowest() noexcept
  {
    return type(__nv_bfloat16_raw{0xff7fu});
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type epsilon() noexcept
  {
    return type(__nv_bfloat16_raw{0x3c00u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type round_error() noexcept
  {
    return type(__nv_bfloat16_raw{0x3f00u});
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
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type infinity() noexcept
  {
    return type(__nv_bfloat16_raw{0x7f80u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type quiet_NaN() noexcept
  {
    return type(__nv_bfloat16_raw{0x7fc0u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type signaling_NaN() noexcept
  {
    return type(__nv_bfloat16_raw{0x7fa0u});
  }
  _LIBCUDACXX_HIDE_FROM_ABI static _LIBCUDACXX_BF16_CONSTEXPR type denorm_min() noexcept
  {
    return type(__nv_bfloat16_raw{0x0001u});
  }

  static constexpr bool is_iec559  = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#  undef _LIBCUDACXX_BF16_CONSTEXPR
#endif // _CCCL_HAS_NVBF16

#if _CCCL_HAS_NVFP8()
template <>
class __numeric_limits_impl<__nv_fp8_e4m3, __numeric_limits_type::__floating_point>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr __nv_fp8_e4m3 __make_value(__nv_fp8_storage_t __val)
  {
#  if defined(_CCCL_BUILTIN_BIT_CAST)
    return _CUDA_VSTD::bit_cast<__nv_fp8_e4m3>(__val);
#  else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ // vvv !_CCCL_BUILTIN_BIT_CAST vvv
    __nv_fp8_e4m3 __ret{};
    __ret.__x = __val;
    return __ret;
#  endif // ^^^ !_CCCL_BUILTIN_BIT_CAST ^^^
  }

public:
  using type = __nv_fp8_e4m3;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 3;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x08u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7eu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0xfeu));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x20u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x30u));
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
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7fu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return type{};
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x01u));
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

template <>
class __numeric_limits_impl<__nv_fp8_e5m2, __numeric_limits_type::__floating_point>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr __nv_fp8_e5m2 __make_value(__nv_fp8_storage_t __val)
  {
#  if defined(_CCCL_BUILTIN_BIT_CAST)
    return _CUDA_VSTD::bit_cast<__nv_fp8_e5m2>(__val);
#  else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ // vvv !_CCCL_BUILTIN_BIT_CAST vvv
    __nv_fp8_e5m2 __ret{};
    __ret.__x = __val;
    return __ret;
#  endif // ^^^ !_CCCL_BUILTIN_BIT_CAST ^^^
  }

public:
  using type = __nv_fp8_e5m2;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 2;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x04u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7bu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0xfbu));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x34u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x38u));
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
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7cu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type quiet_NaN() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7eu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type signaling_NaN() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7du));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type denorm_min() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x01u));
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

#  if _CCCL_CUDACC_AT_LEAST(12, 8)
template <>
class __numeric_limits_impl<__nv_fp8_e8m0, __numeric_limits_type::__floating_point>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr __nv_fp8_e8m0 __make_value(__nv_fp8_storage_t __val)
  {
#    if defined(_CCCL_BUILTIN_BIT_CAST)
    return _CUDA_VSTD::bit_cast<__nv_fp8_e8m0>(__val);
#    else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ // vvv !_CCCL_BUILTIN_BIT_CAST vvv
    __nv_fp8_e8m0 __ret{};
    __ret.__x = __val;
    return __ret;
#    endif // ^^^ !_CCCL_BUILTIN_BIT_CAST ^^^
  }

public:
  using type = __nv_fp8_e8m0;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = false;
  static constexpr int digits       = 0;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 1;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x00u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0xfeu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x00u));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7fu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return __make_value(static_cast<__nv_fp8_storage_t>(0x7fu));
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
    return __make_value(static_cast<__nv_fp8_storage_t>(0xffu));
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
#  endif // _CCCL_CUDACC_AT_LEAST(12, 8)
#endif // _CCCL_HAS_NVFP8()

#if _CCCL_HAS_NVFP6()
template <>
class __numeric_limits_impl<__nv_fp6_e2m3, __numeric_limits_type::__floating_point>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr __nv_fp6_e2m3 __make_value(__nv_fp6_storage_t __val)
  {
#  if defined(_CCCL_BUILTIN_BIT_CAST)
    return _CUDA_VSTD::bit_cast<__nv_fp6_e2m3>(__val);
#  else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ // vvv !_CCCL_BUILTIN_BIT_CAST vvv
    __nv_fp6_e2m3 __ret{};
    __ret.__x = __val;
    return __ret;
#  endif // ^^^ !_CCCL_BUILTIN_BIT_CAST ^^^
  }

public:
  using type = __nv_fp6_e2m3;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 3;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x08u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x1fu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x3fu));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x01u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x04u));
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
    return __make_value(static_cast<__nv_fp6_storage_t>(0x01u));
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

template <>
class __numeric_limits_impl<__nv_fp6_e3m2, __numeric_limits_type::__floating_point>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr __nv_fp6_e3m2 __make_value(__nv_fp6_storage_t __val)
  {
#  if defined(_CCCL_BUILTIN_BIT_CAST)
    return _CUDA_VSTD::bit_cast<__nv_fp6_e3m2>(__val);
#  else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ // vvv !_CCCL_BUILTIN_BIT_CAST vvv
    __nv_fp6_e3m2 __ret{};
    __ret.__x = __val;
    return __ret;
#  endif // ^^^ !_CCCL_BUILTIN_BIT_CAST ^^^
  }

public:
  using type = __nv_fp6_e3m2;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 2;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x04u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x1fu));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x3fu));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x04u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return __make_value(static_cast<__nv_fp6_storage_t>(0x08u));
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
    return __make_value(static_cast<__nv_fp6_storage_t>(0x01u));
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP6()

#if _CCCL_HAS_NVFP4()
template <>
class __numeric_limits_impl<__nv_fp4_e2m1, __numeric_limits_type::__floating_point>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr __nv_fp4_e2m1 __make_value(__nv_fp4_storage_t __val)
  {
#  if defined(_CCCL_BUILTIN_BIT_CAST)
    return _CUDA_VSTD::bit_cast<__nv_fp4_e2m1>(__val);
#  else // ^^^ _CCCL_BUILTIN_BIT_CAST ^^^ // vvv !_CCCL_BUILTIN_BIT_CAST vvv
    __nv_fp4_e2m1 __ret{};
    __ret.__x = __val;
    return __ret;
#  endif // ^^^ !_CCCL_BUILTIN_BIT_CAST ^^^
  }

public:
  using type = __nv_fp4_e2m1;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = true;
  static constexpr int digits       = 1;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 2;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return __make_value(static_cast<__nv_fp4_storage_t>(0x2u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return __make_value(static_cast<__nv_fp4_storage_t>(0x7u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return __make_value(static_cast<__nv_fp4_storage_t>(0xfu));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __FLT_RADIX__;
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type epsilon() noexcept
  {
    return __make_value(static_cast<__nv_fp4_storage_t>(0x1u));
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type round_error() noexcept
  {
    return __make_value(static_cast<__nv_fp4_storage_t>(0x1u));
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
    return __make_value(static_cast<__nv_fp4_storage_t>(0x1u));
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};
#endif // _CCCL_HAS_NVFP4()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_EXT_H
