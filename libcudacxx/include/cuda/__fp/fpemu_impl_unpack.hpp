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

/**
 * @file fpemu_impl_unpack.hpp
 * @brief Common pack/unpack routines for the FPEMU library
 *
 * This header holds the two routines that convert between the packed IEEE-754
 * binary64 representation (fpbits64_t) and the public unpacked ABI
 * (fpbits64_unpacked_t) used by the arithmetic cores:
 *
 *   - impl::__nv_internal_fp64emu_unpack  (packed   -> unpacked)
 *   - impl::__nv_internal_fp64emu_pack    (unpacked -> packed)
 *
 * They are the single, shared prologue/epilogue for every unpacked operation
 * (add / sub / mul / mad / dot / cmul / div / sqrt / cvt / cmp / fma) and every
 * accuracy level (high / mid / low). Because the unpacked approach only crosses the
 * packed<->unpacked boundary outside hot loops, both routines are always the
 * richest, fully-accurate full-range form and are accuracy-INDEPENDENT: denormals
 * are normalized (clz) and inf/nan are encoded in the exponent band on unpack,
 * and the matching full-range epilogue (correctly-rounded by rm, subnormal
 * emission, inf saturation, nan) finalizes every accuracy level on pack. A mid/low core
 * (lower mantissa precision) simply rides on the richer form (subsumption); the
 * accuracy level only affects the precision the core produced, not the range handling
 * here. The packed (legacy non-unified) add/mul/fma kernels do their own inlined
 * lean unpack/pack and do not use these routines.
 *
 * They depend only on the primitives/constants in fpemu_impl_utils.hpp
 * (bit_cast, __round, __fp64_ovfl_sat, __internal_clzll, the FP64_* masks,
 * EXTRA_BITS, BIAS, ...).
 */

#include <cuda/__fp/fpemu_common.hpp>
#include <cuda/__fp/fpemu_impl_utils.hpp>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
{

/**
 * @brief Unpack a packed binary64 value into the public unpacked ABI.
 *
 * Fully-accurate, method-independent, full-range prologue: the sign/exponent/
 * mantissa are extracted, denormals are normalized via clz, and inf/nan are
 * encoded in the exponent band so the matching pack can recover them. The
 * exponent is stored as (biased exponent + 1) with the implicit bit kept in the
 * mantissa (bit 61); pack consumes exactly this convention.
 *
 * @param  x The packed 64-bit value to unpack
 * @return The unpacked representation
 */
__FPEMU_INTERNAL_DECL__
fpbits64_unpacked_t __nv_internal_fp64emu_unpack(fpbits64_t x)
{
    fpbits64_unpacked_t a_unpacked;
    fpemu::uint32x2_t a32 = fpemu::bit_cast<fpemu::uint32x2_t>(x);
    a_unpacked.sign = a32.x[1] & ( 1U << 31 );
    a32.x[1] &= 0x7fffffff;
    int32_t exponent = static_cast<int32_t>(a32.x[1] >> 20);

    // Normalize denormals: leading-zero count of the magnitude (clamped so a
    // normal stays at shift == EXTRA_BITS, and a true zero maps to the zero band).
    uint64_t abs_a = fpemu::bit_cast<uint64_t>(a32);
    int nzeros = fpemu::__internal_clzll(abs_a);
    if (nzeros <  11 ) nzeros = 11;
    if (nzeros == 64 ) nzeros = 2049;
    a32.x[1] = a32.x[1] & 0x000fffff;

    if (exponent == 0x7ff)
    {
        // inf -> 0x00007ff0 band, nan -> 0x0007ff00 band (recovered on pack).
        exponent = (a32.x[1] == 0 && a32.x[0] == 0) ? 0x00007ff0 : 0x0007ff00;
    }
    if (exponent != 0)
    {
        a32.x[1] = a32.x[1] | ( 1 << 20 );   // set the implicit bit
    }
    if (exponent == 0)
    {
        exponent = 12 - nzeros;              // denormal / zero
    }

    int shift = EXTRA_BITS + nzeros - 11;
    uint64_t a64 = fpemu::bit_cast<uint64_t>(a32);

    a_unpacked.exponent = static_cast<uint32_t>(exponent);
    a_unpacked.mantissa = a64 << shift;
    return a_unpacked;
}

/**
 * @brief Pack a public unpacked value back into packed binary64.
 *
 * Fully-accurate, full-range epilogue: subnormal emission, inf/nan
 * classification and correctly-rounded overflow saturation (per rm).
 *
 * It is the exact inverse of unpack (pack(unpack(x)) == x), which the converters
 * (cvt/div/sqrt/cmp) and the unpacked class rely on. unpack stores the *biased*
 * exponent + 1 with the implicit bit kept in the mantissa (bit 61); the proven
 * epilogue expects an exponent one smaller, so we round/place with (exp - 1).
 * The cores emit a matching (exp + 1) so the +1/-1 cancel and op results stay
 * bit-exact while the round-trip identity holds. inf is recovered from the
 * exponent band unpack/cores encode (finite results never reach it); nan is
 * detected from the exponent and wins the overflow branch.
 *
 * @tparam rm Rounding mode
 * @param  x  The unpacked value to pack
 * @return The packed 64-bit value
 */
template<fpemu::rounding rm = fpemu::rounding::def>
__FPEMU_INTERNAL_DECL__
fpbits64_t __nv_internal_fp64emu_pack(fpbits64_unpacked_t x)
{
    const bool    sign     = x.sign != 0;
    const bool    is_inf   = (static_cast<int32_t>(x.exponent) >= 0x2000);
    const int32_t e        = static_cast<int32_t>(x.exponent) - 1;
    int32_t       exponent = e > 0 ? e : 0;

    int shift = e > 0 ? 0 : -e;
#ifndef __CUDA_ARCH__
    shift = (shift > 0) ? (shift > 63) ? 63 : shift : 0;
#endif
    if (shift > 0)
    {
        const uint64_t mask    = (shift >= 64) ? ~0ULL : ((1ULL << shift) - 1);
        const bool     inexact = (x.mantissa & mask) != 0;
        x.mantissa >>= shift;
        if constexpr (rm == fpemu::rounding::rn)
        {
            if (inexact) x.mantissa |= 1;
        }
        else if constexpr (rm == fpemu::rounding::ru)
        {
            if (!sign && inexact) x.mantissa |= 1;
        }
        else if constexpr (rm == fpemu::rounding::rd)
        {
            if (sign && inexact) x.mantissa |= 1;
        }
    }

    fpemu::uint32x2_t mantissa32 = fpemu::bit_cast<fpemu::uint32x2_t>(x.mantissa);
    mantissa32 = fpemu::__round<rm>(mantissa32, 0, sign);

    const bool is_nan = (exponent >= (int)(0x0007ff00 - fpemu::BIAS - 2048 - 1 - 128 + 0xC));

    if (mantissa32.x[0] == 0 && mantissa32.x[1] == 0 && exponent < 0x000007ff)
    {
        exponent = 0;
    }

    if (exponent >= 0x000007ff) exponent = 0x000007ff;

    exponent <<= 20;
    mantissa32.x[1] += exponent;

    if (mantissa32.x[1] >= 0x7ff00000)
    {
        if (is_nan)
        {
            mantissa32.x[0] = 0;
            mantissa32.x[1] = 0x7fffffff;
        }
        else if (is_inf)
        {
            mantissa32.x[0] = 0;
            mantissa32.x[1] = 0x7ff00000;
        }
        else
        {
            int32_t sat_exp = 0;
            fpemu::__fp64_ovfl_sat<rm>(sign, sat_exp, mantissa32);
            mantissa32.x[1] |= (uint32_t)sat_exp << FP64_HI_MANT_SHIFT;
        }
    }

    mantissa32.x[1] += x.sign;
    return fpemu::bit_cast<fpbits64_t>(mantissa32);
}

} // namespace impl


} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_IMPL_UNPACK_H
