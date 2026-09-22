//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_LIMITS_H
#define _CUDA___FP_FPTOOL_LIMITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
// cuda::std::numeric_limits specialization for cuda::experimental::fp_custom
//
// fp_custom<_FpType, _ExpSize, _MantSize> stores a base-type value whose exponent and
// mantissa have been reduced to the requested widths, so the limits below are those of
// the *reduced* format rather than those of the base type. With _ExpSize exponent bits
// the all-ones pattern is spent on infinity and NaN, leaving the usual IEEE-754
// arrangement: a bias of 2^(_ExpSize-1) - 1, a largest finite exponent emax equal to
// that bias, and a smallest normal exponent emin = 1 - bias. With _MantSize stored
// mantissa bits the precision is _MantSize + 1, the implicit leading bit included.
//
// Subnormals depend on the exponent axis alone. The reduction narrows the exponent only
// when _ExpSize is below the base type's own width, and flushes anything that leaves the
// narrowed range to a signed zero, so a reduced exponent has no subnormal range at all.
// At the native exponent width the reduction leaves the exponent untouched and the base
// type's subnormals survive, quantized by the mantissa reduction to steps of
// 2^(emin - _MantSize), which is what denorm_min() reports.
//
// is_iec559 is true only for the fully native format, which is the base type itself in
// every respect. Any reduced format is held in a base-type container rather than in the
// storage its widths describe, and a reduced exponent additionally drops subnormals.
//
// Dynamic sizes: fp_custom_dynamic_size takes a field's width from a runtime setting, so
// there is no compile-time format to describe. Such an instantiation reports
// is_specialized = false, and the remaining members carry the base type's limits so that
// they stay well-formed and readable; they do not describe the format in effect.
*/

#include <cuda/__fp/fptool_custom.h>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief The binary64 bit pattern of 2^__e, subnormal encoding included.
//!
//! @internal Support for the numeric_limits specialization below: the limit values are
//! assembled as bit patterns rather than computed, so that they stay exact under
//! flush-to-zero and remain usable in constant expressions.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint64_t __fptool_limits_pow2_bits(int __e) noexcept
{
  return (__e >= -1022) ? (static_cast<::cuda::std::uint64_t>(__e + 1023) << 52)
                        : (::cuda::std::uint64_t{1} << (__e + 1074));
}

//! @brief The binary64 bit pattern of the largest finite value of a format with the given
//! largest exponent and number of stored mantissa bits.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint64_t
__fptool_limits_max_bits(int __emax, int __mant_bits) noexcept
{
  return (static_cast<::cuda::std::uint64_t>(__emax + 1023) << 52)
       | (((::cuda::std::uint64_t{1} << __mant_bits) - 1) << (52 - __mant_bits));
}
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _FpType, ::cuda::std::uint16_t _ExpSize, ::cuda::std::uint16_t _MantSize>
class numeric_limits<::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>>
{
private:
  using __base   = numeric_limits<_FpType>;
  using __native = ::cuda::experimental::__fp_custom_native_sizes<_FpType>;

  static constexpr ::cuda::std::uint16_t __dyn = ::cuda::experimental::fp_custom_dynamic_size;

  // A runtime-sized field leaves nothing to describe at compile time, so it falls back to
  // the base type's width and is_specialized below reports the values as not meaningful.
  static constexpr int __exp_bits  = (_ExpSize == __dyn) ? int{__native::__exp_size} : int{_ExpSize};
  static constexpr int __mant_bits = (_MantSize == __dyn) ? int{__native::__mant_size} : int{_MantSize};

  // Largest and smallest exponents of a normalized value, the bias being emax itself.
  static constexpr int __emax = (1 << (__exp_bits - 1)) - 1;
  static constexpr int __emin = 1 - __emax;

  // The exponent reduction flushes an out-of-range value to zero, so only a format that
  // keeps the base type's exponent width keeps its subnormals.
  static constexpr bool __has_subnormals = (__exp_bits == int{__native::__exp_size});

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST auto
  __from_bits(::cuda::std::uint64_t __b) noexcept
  {
    return ::cuda::std::bit_cast<::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>>(__b);
  }

public:
  using type = ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>;

  static constexpr bool is_specialized = (_ExpSize != __dyn && _MantSize != __dyn);
  static constexpr bool is_signed      = true;

  static constexpr int digits       = __mant_bits + 1;
  static constexpr int digits10     = (digits - 1) * 30103l / 100000l;
  static constexpr int max_digits10 = 2 + digits * 30103l / 100000l;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type min() noexcept
  {
    return __from_bits(::cuda::experimental::__fptool_limits_pow2_bits(__emin));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type max() noexcept
  {
    return __from_bits(::cuda::experimental::__fptool_limits_max_bits(__emax, __mant_bits));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type lowest() noexcept
  {
    return __from_bits(
      ::cuda::experimental::__fptool_limits_max_bits(__emax, __mant_bits) | (::cuda::std::uint64_t{1} << 63));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __base::radix;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type epsilon() noexcept
  {
    return __from_bits(::cuda::experimental::__fptool_limits_pow2_bits(-__mant_bits));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type round_error() noexcept
  {
    return __from_bits(::cuda::experimental::__fptool_limits_pow2_bits(-1));
  }

  static constexpr int min_exponent   = __emin + 1;
  static constexpr int min_exponent10 = (min_exponent - 1) * 30103l / 100000l;
  static constexpr int max_exponent   = __emax + 1;
  static constexpr int max_exponent10 = max_exponent * 30103l / 100000l;

  static constexpr bool has_infinity      = true;
  static constexpr bool has_quiet_NaN     = true;
  static constexpr bool has_signaling_NaN = true;

  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm =
    __has_subnormals ? denorm_present : denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss = false;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type infinity() noexcept
  {
    return __from_bits(::cuda::std::uint64_t{0x7ff0000000000000});
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type quiet_NaN() noexcept
  {
    return __from_bits(::cuda::std::uint64_t{0x7ff8000000000000});
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type signaling_NaN() noexcept
  {
    return __from_bits(::cuda::std::uint64_t{0x7ff4000000000000});
  }

  // Without a subnormal range the smallest positive value is the smallest normal, which is
  // what the standard asks denorm_min() to report in that case.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type denorm_min() noexcept
  {
    return __has_subnormals ? __from_bits(::cuda::experimental::__fptool_limits_pow2_bits(__emin - __mant_bits))
                            : min();
  }

  // The native format is the base type itself; a reduced one is neither stored nor, once
  // its subnormals are gone, behaved as the format its widths describe.
  static constexpr bool is_iec559 =
    (__exp_bits == int{__native::__exp_size}) && (__mant_bits == int{__native::__mant_size});
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  // Arithmetic is carried out in the base type, so these follow it.
  static constexpr bool traps                    = __base::traps;
  static constexpr bool tinyness_before          = __base::tinyness_before;
  static constexpr float_round_style round_style = round_to_nearest;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_LIMITS_H
