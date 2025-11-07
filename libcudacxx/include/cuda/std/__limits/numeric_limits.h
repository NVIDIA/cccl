//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___LIMITS_NUMERIC_LIMITS_H
#define _CUDA_STD___LIMITS_NUMERIC_LIMITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

enum float_round_style
{
  round_indeterminate       = -1,
  round_toward_zero         = 0,
  round_to_nearest          = 1,
  round_toward_infinity     = 2,
  round_toward_neg_infinity = 3
};

enum _CCCL_DEPRECATED_IN_CXX23 float_denorm_style
{
  denorm_indeterminate = -1,
  denorm_absent        = 0,
  denorm_present       = 1
};

enum class __numeric_limits_type
{
  __integral,
  __bool,
  __floating_point,
  __other,
};

template <class _Tp>
_CCCL_API constexpr __numeric_limits_type __make_numeric_limits_type()
{
  if constexpr (is_same_v<_Tp, bool>)
  {
    return __numeric_limits_type::__bool;
  }
  else if constexpr (is_integral_v<_Tp>)
  {
    return __numeric_limits_type::__integral;
  }
  else if constexpr (__is_fp_v<_Tp>)
  {
    return __numeric_limits_type::__floating_point;
  }
  else
  {
    return __numeric_limits_type::__other;
  }
}

template <class _Tp, __numeric_limits_type = __make_numeric_limits_type<_Tp>()>
class __numeric_limits_impl
{
public:
  using type = _Tp;

  static constexpr bool is_specialized = false;
  _CCCL_API static constexpr type min() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return type();
  }

  static constexpr int digits       = 0;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 0;
  static constexpr bool is_signed   = false;
  static constexpr bool is_integer  = false;
  static constexpr bool is_exact    = false;
  static constexpr int radix        = 0;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type();
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 0;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity                                       = false;
  static constexpr bool has_quiet_NaN                                      = false;
  static constexpr bool has_signaling_NaN                                  = false;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type();
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type();
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = false;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};

template <class _Tp>
class __numeric_limits_impl<_Tp, __numeric_limits_type::__integral>
{
public:
  using type = _Tp;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = type(-1) < type(0);
  static constexpr int digits       = static_cast<int>(sizeof(type) * CHAR_BIT - is_signed);
  static constexpr int digits10     = digits * 3 / 10;
  static constexpr int max_digits10 = 0;
  _CCCL_API static constexpr type min() noexcept
  {
    return static_cast<_Tp>(~max());
  }
  _CCCL_API static constexpr type max() noexcept
  {
    using _Up = make_unsigned_t<_Tp>;
    return static_cast<_Tp>(static_cast<_Up>(~_Up{0}) >> static_cast<int>(is_signed));
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return min();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_exact   = true;
  static constexpr int radix       = 2;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type(0);
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 0;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity                                       = false;
  static constexpr bool has_quiet_NaN                                      = false;
  static constexpr bool has_signaling_NaN                                  = false;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type(0);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = !is_signed;

#if (_CCCL_ARCH(X86_64) && _CCCL_OS(LINUX)) || defined(__pnacl__) || defined(__wasm__)
  static constexpr bool traps = true;
#else
  static constexpr bool traps = false;
#endif
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};

template <>
class __numeric_limits_impl<bool, __numeric_limits_type::__bool>
{
public:
  using type = bool;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = false;
  static constexpr int digits       = 1;
  static constexpr int digits10     = 0;
  static constexpr int max_digits10 = 0;

  _CCCL_API static constexpr type min() noexcept
  {
    return false;
  }
  _CCCL_API static constexpr type max() noexcept
  {
    return true;
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return min();
  }

  static constexpr bool is_integer = true;
  static constexpr bool is_exact   = true;
  static constexpr int radix       = 2;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type(0);
  }

  static constexpr int min_exponent   = 0;
  static constexpr int min_exponent10 = 0;
  static constexpr int max_exponent   = 0;
  static constexpr int max_exponent10 = 0;

  static constexpr bool has_infinity                                       = false;
  static constexpr bool has_quiet_NaN                                      = false;
  static constexpr bool has_signaling_NaN                                  = false;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type(0);
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type(0);
  }

  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_toward_zero;
};

template <class _Tp>
class __numeric_limits_impl<_Tp, __numeric_limits_type::__floating_point>
{
  static constexpr auto __fmt = __fp_format_of_v<_Tp>;

  [[nodiscard]] _CCCL_API static constexpr int __max_exponent10() noexcept
  {
    switch (__fmt)
    {
      case __fp_format::__binary16:
        return 4;
      case __fp_format::__binary32:
        return 38;
      case __fp_format::__binary64:
        return 308;
      case __fp_format::__binary128:
        return 4932;
      case __fp_format::__bfloat16:
        return 38;
      case __fp_format::__fp80_x86:
        return 4932;
      case __fp_format::__fp8_nv_e4m3:
        return 2;
      case __fp_format::__fp8_nv_e5m2:
        return 4;
      case __fp_format::__fp8_nv_e8m0:
        return 38;
      case __fp_format::__fp6_nv_e2m3:
        return 0;
      case __fp_format::__fp6_nv_e3m2:
        return 1;
      case __fp_format::__fp4_nv_e2m1:
        return 0;
      default:
        return 0;
    }
  }

public:
  using type = _Tp;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed   = __fp_is_signed_v<__fmt>;
  static constexpr int digits       = __fp_digits_v<__fmt>;
  static constexpr int digits10     = static_cast<int>(((digits - 1) * 30103ll) / 100000ll);
  static constexpr int max_digits10 = 2 + (digits * 30103ll) / 100000ll;
  [[nodiscard]] _CCCL_API static constexpr type min() noexcept
  {
    return ::cuda::std::__fp_min<_Tp>();
  }
  [[nodiscard]] _CCCL_API static constexpr type max() noexcept
  {
    return ::cuda::std::__fp_max<_Tp>();
  }
  [[nodiscard]] _CCCL_API static constexpr type lowest() noexcept
  {
    if constexpr (is_signed)
    {
      return ::cuda::std::__fp_neg(::cuda::std::__fp_max<_Tp>());
    }
    else if constexpr (__fmt != __fp_format::__fp8_nv_e8m0)
    {
      return ::cuda::std::__fp_zero<_Tp>();
    }
    else
    {
      return ::cuda::std::__fp_min<_Tp>();
    }
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = 2;
  [[nodiscard]] _CCCL_API static constexpr type epsilon() noexcept
  {
    return ::cuda::std::__fp_epsilon<_Tp>();
  }
  [[nodiscard]] _CCCL_API static constexpr type round_error() noexcept
  {
    // 1.0 for nvfp8_e8m0, 0.5 for all other types
    if constexpr (__fmt == __fp_format::__fp8_nv_e8m0)
    {
      return ::cuda::std::__fp_one<_Tp>();
    }
    else if constexpr (__fp_is_native_type_v<_Tp>)
    {
      return static_cast<_Tp>(0.5f);
    }
    else if (__fmt == __fp_format::__fp4_nv_e2m1)
    {
      return ::cuda::std::__fp_min<_Tp>();
    }
    else
    {
      using _Storage = __fp_storage_t<__fmt>;
      return static_cast<_Storage>((static_cast<_Storage>(__fp_exp_bias_v<__fmt> - 1) << __fp_mant_nbits_v<__fmt>)
                                   | __fp_explicit_bit_mask_v<__fmt>);
    }
  }

  static constexpr int min_exponent   = __fp_exp_min_v<__fmt> + 1;
  static constexpr int min_exponent10 = (30103ll * __fp_exp_min_v<__fmt>) / 100000ll;
  static constexpr int max_exponent   = __fp_exp_max_v<__fmt> + 1;
  static constexpr int max_exponent10 = __max_exponent10();

  static constexpr bool has_infinity      = __fp_has_inf_v<__fmt>;
  static constexpr bool has_quiet_NaN     = __fp_has_nan_v<__fmt>;
  static constexpr bool has_signaling_NaN = __fp_has_nans_v<__fmt>;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm =
    (__fmt != __fp_format::__fp8_nv_e8m0) ? denorm_present : denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss = false;

  [[nodiscard]] _CCCL_API static constexpr type infinity() noexcept
  {
    if constexpr (has_infinity)
    {
      return ::cuda::std::__fp_inf<_Tp>();
    }
    else
    {
      return _Tp{};
    }
  }

  [[nodiscard]] _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    if constexpr (has_quiet_NaN)
    {
      return ::cuda::std::__fp_nan<_Tp>();
    }
    else
    {
      return _Tp{};
    }
  }

  [[nodiscard]] _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    if constexpr (has_signaling_NaN)
    {
      return ::cuda::std::__fp_nans<_Tp>();
    }
    else
    {
      return _Tp{};
    }
  }

  [[nodiscard]] _CCCL_API static constexpr type denorm_min() noexcept
  {
    if constexpr (has_denorm)
    {
      return ::cuda::std::__fp_denorm_min<_Tp>();
    }
    else
    {
      return _Tp{};
    }
  }

  static constexpr bool is_iec559 =
    __fmt == __fp_format::__binary16 || __fmt == __fp_format::__binary32 || __fmt == __fp_format::__binary64
    || __fmt == __fp_format::__binary128 || __fmt == __fp_format::__bfloat16 || __fmt == __fp_format::__fp80_x86;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps           = false;
  static constexpr bool tinyness_before = false;
  static constexpr float_round_style round_style =
    (__fmt != __fp_format::__fp8_nv_e8m0) ? round_to_nearest : round_toward_zero;
};

template <class _Tp>
class numeric_limits : public __numeric_limits_impl<_Tp>
{};

template <class _Tp>
class numeric_limits<const _Tp> : public numeric_limits<_Tp>
{};

template <class _Tp>
class numeric_limits<volatile _Tp> : public numeric_limits<_Tp>
{};

template <class _Tp>
class numeric_limits<const volatile _Tp> : public numeric_limits<_Tp>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___LIMITS_NUMERIC_LIMITS_H
