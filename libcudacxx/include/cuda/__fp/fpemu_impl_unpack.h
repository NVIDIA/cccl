//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_UNPACK_H
#define _CUDA___FP_FPEMU_IMPL_UNPACK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fpemu_impl_unpack.h
//! @brief Common pack/unpack routines for the FPEMU library
//!
//! This header holds the two routines that convert between the packed IEEE-754
//! binary64 representation (__fpbits64) and the public unpacked ABI
//! (__fpbits64_unpacked) used by the arithmetic cores:
//!
//!   - __internal_fp64emu_unpack  (packed   -> unpacked)
//!   - __internal_fp64emu_pack    (unpacked -> packed)
//!
//! They are the single, shared prologue/epilogue for every unpacked operation
//! (add / sub / mul / mad / dot / cmul / div / sqrt / cvt / cmp / fma) and every
//! accuracy level (high / mid / low). Because the unpacked approach only crosses the
//! packed<->unpacked boundary outside hot loops, both routines are always the
//! richest, fully-accurate full-range form and are accuracy-INDEPENDENT: denormals
//! are normalized (clz) and inf/nan are encoded in the exponent band on unpack,
//! and the matching full-range epilogue (correctly-rounded by rm, subnormal
//! emission, inf saturation, nan) finalizes every accuracy level on pack. A mid/low core
//! (lower mantissa precision) simply rides on the richer form (subsumption); the
//! accuracy level only affects the precision the core produced, not the range handling
//! here. The packed (legacy non-unified) add/mul/fma kernels do their own inlined
//! lean unpack/pack and do not use these routines.
//!
//! They depend only on the primitives/constants in fpemu_impl.h
//! (bit_cast, __round, __fp64_ovfl_sat, ::cuda::std::countl_zero, the FP64_* masks,
//! EXTRA_BITS, BIAS, ...).

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/std/__bit/countl.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Unpack a packed binary64 value into the public unpacked ABI.
//!
//! Fully-accurate, method-independent, full-range prologue: the sign/exponent/
//! mantissa are extracted, denormals are normalized via clz, and inf/nan are
//! encoded in the exponent band so the matching pack can recover them. The
//! exponent is stored as (biased exponent + 1) with the implicit bit kept in the
//! mantissa (bit 61); pack consumes exactly this convention.
//!
//! @param  x The packed 64-bit value to unpack
//! @return The unpacked representation
_CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_unpack(__fpbits64 __x) noexcept
{
  __fpbits64_unpacked __a_unpacked;
  __uint32x2 __a32  = __fpemu_bit_cast<__uint32x2>(__x);
  __a_unpacked.sign = __a32.x[1] & (1U << 31);
  __a32.x[1] &= 0x7fffffff;
  int32_t __exponent = static_cast<int32_t>(__a32.x[1] >> 20);

  // Normalize denormals: leading-zero count of the magnitude (clamped so a
  // normal stays at shift == EXTRA_BITS, and a true zero maps to the zero band).
  uint64_t __abs_a = __fpemu_bit_cast<uint64_t>(__a32);
  int __nzeros     = ::cuda::std::countl_zero(__abs_a);
  if (__nzeros < 11)
  {
    __nzeros = 11;
  }
  if (__nzeros == 64)
  {
    __nzeros = 2049;
  }
  __a32.x[1] = __a32.x[1] & 0x000fffff;

  if (__exponent == 0x7ff)
  {
    // inf -> 0x00007ff0 band, nan -> 0x0007ff00 band (recovered on pack).
    __exponent = (__a32.x[1] == 0 && __a32.x[0] == 0) ? 0x00007ff0 : 0x0007ff00;
  }
  if (__exponent != 0)
  {
    __a32.x[1] = __a32.x[1] | (1 << 20); // set the implicit bit
  }
  if (__exponent == 0)
  {
    __exponent = 12 - __nzeros; // denormal / zero
  }

  int __shift    = EXTRA_BITS + __nzeros - 11;
  uint64_t __a64 = __fpemu_bit_cast<uint64_t>(__a32);

  __a_unpacked.exponent = static_cast<uint32_t>(__exponent);
  __a_unpacked.mantissa = __a64 << __shift;
  return __a_unpacked;
}

//! @brief Pack a public unpacked value back into packed binary64.
//!
//! Fully-accurate, full-range epilogue: subnormal emission, inf/nan
//! classification and correctly-rounded overflow saturation (per rm).
//!
//! It is the exact inverse of unpack (pack(unpack(x)) == x), which the converters
//! (cvt/div/sqrt/cmp) and the unpacked class rely on. unpack stores the *biased*
//! exponent + 1 with the implicit bit kept in the mantissa (bit 61); the proven
//! epilogue expects an exponent one smaller, so we round/place with (exp - 1).
//! The cores emit a matching (exp + 1) so the +1/-1 cancel and op results stay
//! bit-exact while the round-trip identity holds. inf is recovered from the
//! exponent band unpack/cores encode (finite results never reach it); nan is
//! detected from the exponent and wins the overflow branch.
//!
//! @tparam rm Rounding mode
//! @param  x  The unpacked value to pack
//! @return The packed 64-bit value
template <__fpemu_rounding _Rm = __fpemu_rounding::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_pack(__fpbits64_unpacked __x) noexcept
{
  const bool __sign   = __x.sign != 0;
  const bool __is_inf = (static_cast<int32_t>(__x.exponent) >= 0x2000);
  const int32_t __e   = static_cast<int32_t>(__x.exponent) - 1;
  int32_t __exponent  = __e > 0 ? __e : 0;

  int __shift = __e > 0 ? 0 : -__e;
#ifndef __CUDA_ARCH__
  __shift = (__shift > 0) ? (__shift > 63) ? 63 : __shift : 0;
#endif
  if (__shift > 0)
  {
    const uint64_t __mask = (__shift >= 64) ? ~0ULL : ((1ULL << __shift) - 1);
    const bool __inexact  = (__x.mantissa & __mask) != 0;
    __x.mantissa >>= __shift;
    if constexpr (_Rm == __fpemu_rounding::rn)
    {
      if (__inexact)
      {
        __x.mantissa |= 1;
      }
    }
    else if constexpr (_Rm == __fpemu_rounding::ru)
    {
      if (!__sign && __inexact)
      {
        __x.mantissa |= 1;
      }
    }
    else if constexpr (_Rm == __fpemu_rounding::rd)
    {
      if (__sign && __inexact)
      {
        __x.mantissa |= 1;
      }
    }
  }

  __uint32x2 __mantissa32 = __fpemu_bit_cast<__uint32x2>(__x.mantissa);
  __mantissa32            = __round<_Rm>(__mantissa32, 0, __sign);

  const bool __is_nan = (__exponent >= (int) (0x0007ff00 - __fpemu_bias - 2048 - 1 - 128 + 0xC));

  if (__mantissa32.x[0] == 0 && __mantissa32.x[1] == 0 && __exponent < 0x000007ff)
  {
    __exponent = 0;
  }

  if (__exponent >= 0x000007ff)
  {
    __exponent = 0x000007ff;
  }

  __exponent <<= 20;
  __mantissa32.x[1] += __exponent;

  if (__mantissa32.x[1] >= 0x7ff00000)
  {
    if (__is_nan)
    {
      __mantissa32.x[0] = 0;
      __mantissa32.x[1] = 0x7fffffff;
    }
    else if (__is_inf)
    {
      __mantissa32.x[0] = 0;
      __mantissa32.x[1] = 0x7ff00000;
    }
    else
    {
      int32_t __sat_exp = 0;
      __fp64_ovfl_sat<_Rm>(__sign, __sat_exp, __mantissa32);
      __mantissa32.x[1] |= (uint32_t) __sat_exp << _CCCL_FP64_HI_MANT_SHIFT;
    }
  }

  __mantissa32.x[1] += __x.sign;
  return __fpemu_bit_cast<__fpbits64>(__mantissa32);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_IMPL_UNPACK_H
