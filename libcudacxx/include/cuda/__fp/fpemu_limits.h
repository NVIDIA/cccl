//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_LIMITS_H
#define _CUDA___FP_FPEMU_LIMITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
// cuda::std::numeric_limits specializations for cuda::experimental::fpemu and
// cuda::experimental::fpemu_unpacked
//
// fpemu is bit-identical to the base type it emulates, so its limits are the base type's,
// reported through the emulated type. Two properties depend on the accuracy level rather
// than on the storage:
//   - has_denorm. Only fpemu_accuracy::high covers the full IEEE-754 range; mid and low
//     work in the normal range, so a subnormal does not survive their arithmetic and the
//     smallest positive value those levels reach is min().
//   - is_iec559. It asks for correctly rounded operations over that full range, which
//     again is high alone. The storage is a base-type bit pattern at every level, so the
//     representation-level members (has_infinity, has_quiet_NaN, has_signaling_NaN) stay
//     true throughout and describe what the type can hold, not what mid and low produce.
//
// fpemu_unpacked keeps the value in separate sign/exponent/mantissa fields, with
// _CCCL_FPEMU_EXTRA_BITS guard bits below the base significand. Its precision is therefore
// the base type's plus those guard bits -- 62 bits for binary64 -- and digits, epsilon(),
// max() and denorm_min() all report that wider format rather than the storage format the
// value takes when it is packed. This is the same reading the class documentation gives
// for why an unpacked chain is more accurate than the same chain in double, and it is the
// reading numeric_limits<fpmp2> already takes in reporting the double-word precision
// rather than its component's.
//
// A consequence worth stating: max() exceeds the base type's max, so packing it overflows
// to infinity, exactly as converting long double's max to double does. The exponent range
// is unchanged from the base type, being what the cores and the pack epilogue agree on.
//
// The encoding follows __internal_fp64emu_unpack: the exponent field holds the biased
// base-type exponent and the mantissa holds the significand with its implicit bit at
// bit (base mantissa width + _CCCL_FPEMU_EXTRA_BITS), so that a value is
// mantissa * 2^(exponent - bias - implicit bit position). Subnormals are normalized into
// that form, which drives the exponent field negative and is why it is stored unsigned.
*/

#include <cuda/__fp/fpemu.h>
#include <cuda/__fp/fpemu_impl.h>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Position of the implicit significand bit in the unpacked mantissa field.
//!
//! @internal Support for the numeric_limits specializations below.
inline constexpr int __fpemu_limits_implicit_bit = 52 + _CCCL_FPEMU_EXTRA_BITS;

//! @brief The unpacked encoding of 2^__e.
//!
//! @internal The limit values are assembled as field patterns rather than computed, so
//! that they stay exact under flush-to-zero and remain usable in constant expressions.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __fpbits64_unpacked __fpemu_limits_pow2(int __e) noexcept
{
  return __fpbits64_unpacked{
    0u, static_cast<::cuda::std::uint32_t>(__e + 1023), ::cuda::std::uint64_t{1} << __fpemu_limits_implicit_bit};
}

//! @brief The unpacked encoding of the NaN with the given packed bit pattern.
//!
//! @internal The payload is taken from the base type rather than assumed, so that the
//! reported NaN is the one numeric_limits gives for the base type on the platform at
//! hand. unpack leaves the payload in place, sets the implicit bit and shifts by the
//! guard bits, encoding NaN in an exponent band of its own.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __fpbits64_unpacked
__fpemu_limits_nan(::cuda::std::uint64_t __bits) noexcept
{
  constexpr ::cuda::std::uint64_t __mant_mask = (::cuda::std::uint64_t{1} << 52) - 1;
  constexpr ::cuda::std::uint64_t __implicit  = ::cuda::std::uint64_t{1} << 52;
  constexpr ::cuda::std::uint32_t __nan_band  = 0x0007ff00u;

  return __fpbits64_unpacked{static_cast<::cuda::std::uint32_t>(__bits >> 63) << 31,
                             __nan_band,
                             ((__bits & __mant_mask) | __implicit) << _CCCL_FPEMU_EXTRA_BITS};
}
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//==============================================================================
// fpemu: the base type's limits, reported through the emulated type
//==============================================================================
template <class _FpType, ::cuda::experimental::fpemu_accuracy _Met>
class numeric_limits<::cuda::experimental::fpemu<_FpType, _Met>>
{
private:
  using __base = numeric_limits<_FpType>;

  // Only the full-range level covers subnormals; mid and low work in the normal range.
  static constexpr bool __is_full_range = (_Met == ::cuda::experimental::fpemu_accuracy::high);

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST auto __from(_FpType __v) noexcept
  {
    return ::cuda::std::bit_cast<::cuda::experimental::fpemu<_FpType, _Met>>(__v);
  }

public:
  using type = ::cuda::experimental::fpemu<_FpType, _Met>;

  static constexpr bool is_specialized = true;
  static constexpr bool is_signed      = true;

  static constexpr int digits       = __base::digits;
  static constexpr int digits10     = __base::digits10;
  static constexpr int max_digits10 = __base::max_digits10;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type min() noexcept
  {
    return __from(__base::min());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type max() noexcept
  {
    return __from(__base::max());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type lowest() noexcept
  {
    return __from(__base::lowest());
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __base::radix;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type epsilon() noexcept
  {
    return __from(__base::epsilon());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type round_error() noexcept
  {
    return __from(__base::round_error());
  }

  static constexpr int min_exponent   = __base::min_exponent;
  static constexpr int min_exponent10 = __base::min_exponent10;
  static constexpr int max_exponent   = __base::max_exponent;
  static constexpr int max_exponent10 = __base::max_exponent10;

  // The storage is a base-type bit pattern at every accuracy level, so it can hold each
  // of these whatever the level's arithmetic does with them.
  static constexpr bool has_infinity      = true;
  static constexpr bool has_quiet_NaN     = true;
  static constexpr bool has_signaling_NaN = true;

  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm =
    __is_full_range ? denorm_present : denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss = false;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type infinity() noexcept
  {
    return __from(__base::infinity());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type quiet_NaN() noexcept
  {
    return __from(__base::quiet_NaN());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type signaling_NaN() noexcept
  {
    return __from(__base::signaling_NaN());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type denorm_min() noexcept
  {
    return __is_full_range ? __from(__base::denorm_min()) : min();
  }

  // Correct rounding over the full range is what the high level provides and what the
  // other two trade away.
  static constexpr bool is_iec559  = __is_full_range;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = __base::traps;
  static constexpr bool tinyness_before          = __base::tinyness_before;
  static constexpr float_round_style round_style = round_to_nearest;
};

//==============================================================================
// fpemu_unpacked: the base type's exponent range at the guard-bit precision
//==============================================================================
template <class _FpType, ::cuda::experimental::fpemu_accuracy _Met>
class numeric_limits<::cuda::experimental::fpemu_unpacked<_FpType, _Met>>
{
private:
  using __base = numeric_limits<_FpType>;

  static constexpr bool __is_full_range = (_Met == ::cuda::experimental::fpemu_accuracy::high);

  static constexpr int __implicit_bit = ::cuda::experimental::__fpemu_limits_implicit_bit;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST auto
  __from(::cuda::experimental::__fpbits64_unpacked __v) noexcept
  {
    return ::cuda::std::bit_cast<::cuda::experimental::fpemu_unpacked<_FpType, _Met>>(__v);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST auto __pow2(int __e) noexcept
  {
    return __from(::cuda::experimental::__fpemu_limits_pow2(__e));
  }

public:
  using type = ::cuda::experimental::fpemu_unpacked<_FpType, _Met>;

  static constexpr bool is_specialized = true;
  static constexpr bool is_signed      = true;

  // The guard bits below the base significand are carried by every unpacked value, so
  // they are part of the precision the type offers rather than an internal detail.
  static constexpr int digits       = __base::digits + _CCCL_FPEMU_EXTRA_BITS;
  static constexpr int digits10     = (digits - 1) * 30103l / 100000l;
  static constexpr int max_digits10 = 2 + digits * 30103l / 100000l;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type min() noexcept
  {
    return __pow2(__base::min_exponent - 1);
  }

  // The largest finite value: every significand bit set at the largest exponent, which is
  // above the base type's max and packs to infinity.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type max() noexcept
  {
    return __from(::cuda::experimental::__fpbits64_unpacked{
      0u,
      static_cast<::cuda::std::uint32_t>(__base::max_exponent - 1 + 1023),
      (::cuda::std::uint64_t{1} << (__implicit_bit + 1)) - 1});
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type lowest() noexcept
  {
    return __from(::cuda::experimental::__fpbits64_unpacked{
      1u << 31,
      static_cast<::cuda::std::uint32_t>(__base::max_exponent - 1 + 1023),
      (::cuda::std::uint64_t{1} << (__implicit_bit + 1)) - 1});
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __base::radix;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type epsilon() noexcept
  {
    return __pow2(1 - digits);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type round_error() noexcept
  {
    return __pow2(-1);
  }

  // The exponent range is the base type's: it is what the cores and the pack epilogue
  // agree on, the extra bits buying precision rather than range.
  static constexpr int min_exponent   = __base::min_exponent;
  static constexpr int min_exponent10 = __base::min_exponent10;
  static constexpr int max_exponent   = __base::max_exponent;
  static constexpr int max_exponent10 = __base::max_exponent10;

  static constexpr bool has_infinity      = true;
  static constexpr bool has_quiet_NaN     = true;
  static constexpr bool has_signaling_NaN = true;

  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm =
    __is_full_range ? denorm_present : denorm_absent;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss = false;

  // Unpack encodes infinity and NaN in exponent bands of their own, which the matching
  // pack recovers; these are the patterns it produces.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type infinity() noexcept
  {
    return __from(
      ::cuda::experimental::__fpbits64_unpacked{0u, 0x00007ff0u, ::cuda::std::uint64_t{1} << __implicit_bit});
  }

  // The payloads come from the base type, whose choice of them is implementation-defined.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type quiet_NaN() noexcept
  {
    return __from(
      ::cuda::experimental::__fpemu_limits_nan(::cuda::std::bit_cast<::cuda::std::uint64_t>(__base::quiet_NaN())));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type signaling_NaN() noexcept
  {
    return __from(
      ::cuda::experimental::__fpemu_limits_nan(::cuda::std::bit_cast<::cuda::std::uint64_t>(__base::signaling_NaN())));
  }

  // A subnormal is normalized on the way in, so the smallest positive value is the base
  // type's smallest subnormal carried in normalized form.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static _CCCL_CONSTEXPR_BIT_CAST type denorm_min() noexcept
  {
    return __is_full_range ? __pow2(__base::min_exponent - __base::digits) : min();
  }

  // A significand wider than the format it is stored in is not an IEEE-754 format.
  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = __base::traps;
  static constexpr bool tinyness_before          = __base::tinyness_before;
  static constexpr float_round_style round_style = round_to_nearest;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_LIMITS_H
