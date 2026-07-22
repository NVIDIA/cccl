//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_FMA_H
#define _CUDA___FP_FPEMU_IMPL_FMA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fpemu_impl_fma.h
//! @brief Implementation of fused multiply-add operations (FMA & MAD) for FPEMU floating point emulation library
//!
//! This header provides the implementation of fused multiply-add operations for the FPEMU library.
//! It includes:
//!   - Fused multiply-add functions for different accuracy and range configurations
//!   - Special case handling for NaN, inf, zero, etc
//!
//! The implementation is designed to work across both host and device code
//! through appropriate decorators and provide bit-exact results matching hardware
//! floating point units.

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/std/__bit/countl.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined(__CUDA_ARCH__))
// Host seed: the libm symbol. Declared noexcept to match the standard <math.h>
// prototype (glibc marks it __THROW); otherwise the extern-"C" redeclaration
// conflicts with ::fma when <cmath> is also in the TU (-Werror).
extern "C" double fma(double __x, double __y, double __z) noexcept;
#endif

//! @brief Pure FMA core operating on the unpacked representation.
//!
//! Consumes/produces __fpbits64_unpacked exactly as produced by the universal
//! __internal_fp64emu_unpack and consumed by __internal_fp64emu_pack.
//! Inputs carry a normalized mantissa (implicit bit set, denormals normalized)
//! with inf/nan encoded in the exponent band (around the 0x00007ff0 / 0x0007ff00
//! magics), so the floating-point class is read from the exponent rather than a
//! separate field. The returned value is the pre-rounding intermediate: a 64-bit
//! mantissa with a sticky LSB and an exponent that may be <= 0 (subnormal) or in
//! the inf/nan band. Final rounding, subnormal shifting and inf/saturate are the
//! job of pack. Templated on the rounding mode (sign-of-zero, rd handling) and
//! the method (product accuracy via __mul_128 and range-based special handling).
template <fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64_unpacked
__internal_fp64emu_fma_unpacked(__fpbits64_unpacked __a, __fpbits64_unpacked __b, __fpbits64_unpacked __c) noexcept
{
  constexpr fpemu_accuracy __acc_forced = fpemu_accuracy::_CCCL_FPEMU_FMA_METHOD;
  constexpr fpemu_accuracy __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;
  // Unpacked cores always run on the fully-accurate full-range unpack/pack
  // boundary (method-independent): the inf/nan folds stay live and underflow
  // flows to the full pack (no FTZ, correct subnormal + min_normal round-up).

  // Inf/Nan exponent magics produced by the universal unpack.
  constexpr uint32_t __inf_exp = 0x00007ff0u;
  constexpr int32_t __nan_exp  = 0x0007ff00;

  __fpbits64_unpacked __r;
  __fpemu_uint128 __mantissa_ab;
  __uint64x2 __ab_res;

  // MUL START:
  // Compute mantissa_ab - the product of a and b in 128-bit
  __uint32x2 __a_32x2 = ::cuda::std::bit_cast<__uint32x2>(__a.mantissa);
  __uint32x2 __b_32x2 = ::cuda::std::bit_cast<__uint32x2>(__b.mantissa);
  __ab_res            = ::cuda::std::bit_cast<__uint64x2>(__mul_128<__acc_used>(__a_32x2, __b_32x2));

  __mantissa_ab              = ::cuda::std::bit_cast<__fpemu_uint128>(__ab_res);
  __uint32x4 __mantissa_ab32 = ::cuda::std::bit_cast<__uint32x4>(__mantissa_ab);

  // Exponents/signs (read with explicit signedness: the public field is
  // uint32 but the core needs signed arithmetic for subnormal exponents).
  int32_t __exponent_ab = (int32_t) __a.exponent + (int32_t) __b.exponent - (int32_t) __fpemu_bias;
  int32_t __exponent_c  = (int32_t) __c.exponent;
  int32_t __sign_ab     = (int32_t) (__a.sign ^ __b.sign);
  int32_t __sign_c      = (int32_t) __c.sign;

  // Compute exponent_ab_new - the exponent of the product of a and b
  int __mul_nzeros          = __mantissa_ab32.hi.x[1] < 0x08000000;
  int32_t __exponent_ab_new = __exponent_ab - __mul_nzeros + 1;
  // Shift mantissa_ab
  __mantissa_ab = __mantissa_ab << (11 - EXTRA_BITS + __mul_nzeros);
  // Compute mantissa_c - the mantissa of c
  __fpemu_uint128 __mantissa_c = __c.mantissa;
  // Compute mantissa_r - the result of the product of a and b and c
  __fpemu_uint128 __mantissa_r;

  {
    // Check if a or b is inf and c is inf and sign_ab != sign_c then return NaN
    if ((__a.exponent == __inf_exp || __b.exponent == __inf_exp) && __c.exponent == __inf_exp && __sign_ab != __sign_c)
    {
      __exponent_ab_new = __nan_exp;
    }
  }

  // Check if exponent_ab_new is INF_ZERO then return NaN
  if (__exponent_ab_new == (int32_t) __fpemu_inf_zero)
  {
    __exponent_ab_new = __nan_exp;
  }

  // ADD START:
  //  Compute exponent_r - the larger of exponent_ab_new and exponent_c
  int32_t __exponent_r = _CCCL_FPEMU_MAX(__exponent_ab_new, __exponent_c);

  // Compute delta_a and delta_b for mantissas shift
  int32_t __delta_a = __exponent_r - __exponent_ab_new;
  int32_t __delta_b = __exponent_r - __exponent_c;

#ifndef __CUDA_ARCH__
  __delta_a = (__delta_a > 127) ? 127 : __delta_a;
  __delta_b = (__delta_b > 127) ? 127 : __delta_b;
#endif
  // Shift mantissas with jam only (SoftFloat shiftRightJam*); round at pack
  __mantissa_ab = __shr_128_jam(__mantissa_ab, __delta_a);
  __mantissa_c  = __shr_128_jam(__mantissa_c << 64, __delta_b);

  // Add or subtract mantissas
  uint32_t __sign_r = __sign_ab;
  if (__sign_ab == __sign_c)
  {
    __mantissa_r = __mantissa_ab + __mantissa_c;
  }
  else if (__mantissa_ab == __mantissa_c)
  {
    __mantissa_r = 0;
  }
  else if ((__mantissa_ab > __mantissa_c))
  {
    __mantissa_r = __mantissa_ab - __mantissa_c;
  }
  else // mantissa_ab < mantissa_c
  {
    __sign_r     = __sign_c;
    __mantissa_r = __mantissa_c - __mantissa_ab;
  }

  if (__mantissa_r == 0)
  {
    // Exact cancellation -> zero. IEEE-754 6.3 would make this -0 under
    // round-toward-negative (rd); that rounding-dependent zero sign is
    // intentionally NOT honored here (the core is rounding-independent), so
    // the zero is -0 only when both effective signs are negative.
    __sign_r = __sign_ab & __sign_c;
  }

  // Normalize mantissa_r
  // use reinterpret_cast to avoid slowdown from bit_cast
  uint64_t* __m = reinterpret_cast<uint64_t*>(&__mantissa_r);
  int __nzeros  = (__m[1] == 0) ? (::cuda::std::countl_zero((uint64_t) (__m[0] + 64)))
                                : (::cuda::std::countl_zero((uint64_t) (__m[1] << 1)));

  // Shift mantissa_r
  __mantissa_r = (__nzeros == 0) ? (__mantissa_r >> 1) : (__mantissa_r << (__nzeros - 1));

  __uint64x2 __mantissa_r64 = ::cuda::std::bit_cast<__uint64x2>(__mantissa_r);

  // The result class (inf vs finite-overflow) is recoverable from the
  // exponent band by pack: genuine infinities inherit the huge exponent of
  // their inf operand, finite results never reach it. So no class field is
  // written here; the inf-inf -> NaN and inf*0 -> NaN cases were already
  // folded into exponent_ab_new (NAN_EXP) above.
  __r.sign = __sign_r;
  // +1 matches the unified pack's "mask the implicit bit" convention
  // (pack reconstitutes via exp-1); the +1/-1 cancel so packed FMA is
  // bit-exact with the legacy add-convention packer.
  __r.exponent = static_cast<uint32_t>(__exponent_r - __nzeros + 1);
  __r.mantissa = __mantissa_r64.x[1] | (__mantissa_r64.x[0] != 0);

  // The unpacked core runs on the full-range boundary: underflow (and the rare
  // top-subnormal -> min_normal round-up) flows to the full pack, which has the
  // complete mantissa and resolves the correct subnormal / min_normal for every
  // rounding mode. No FTZ here, so no rounding-dependent fix-up is needed.

  return __r;
} // __internal_fp64emu_fma_unpacked

template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_fma(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  // Forced parameters for the fused multiply-add operation
  constexpr fpemu_accuracy __acc_forced = fpemu_accuracy::_CCCL_FPEMU_FMA_METHOD;
  constexpr fpemu_accuracy __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;

  {
    {
      // FMA = pack(fma_unpacked(unpack(x), unpack(y), unpack(z))). The fma_unpacked
      // core selects accurate/def/fast internally; the universal full-range
      // unpack/pack are the shared prologue/epilogue (def/fast are full-range here).
      __fpbits64_unpacked __a = __internal_fp64emu_unpack(__x);
      __fpbits64_unpacked __b = __internal_fp64emu_unpack(__y);
      __fpbits64_unpacked __c = __internal_fp64emu_unpack(__z);
      __fpbits64_unpacked __r = __internal_fp64emu_fma_unpacked<__acc_used>(__a, __b, __c);
      __fpbits64 __result     = __internal_fp64emu_pack<_Rm>(__r);

      if constexpr (_Rm == __fpemu_rounding::rd)
      {
        // Exact cancellation (a*b + c == 0 with opposite effective signs)
        // must yield -0 under round-toward-negative. The rounding-independent
        // core packs an exact zero to +0 (r.mantissa == 0); a misaligned
        // remainder can also surface as a tiny artifact. Both map to -0 here.
        // A genuine underflow keeps a nonzero core mantissa and stays +0.
        const bool __opposite_signs = ((__a.sign ^ __b.sign) != __c.sign);
        const bool __exact_zero     = (__r.mantissa == 0) && ((__result << 1) == 0);
        const bool __tiny_artifact  = (__result == UINT64_C(0x0000000100000000));
        if (__opposite_signs && (__exact_zero || __tiny_artifact))
        {
          __fpbits64_unpacked __zneg;
          __zneg.sign     = 1U << 31;
          __zneg.exponent = 0;
          __zneg.mantissa = 0;
          __result        = __internal_fp64emu_pack<_Rm>(__zneg);
        }
      }
      return __result;
    }
  }
} // __internal_fp64emu_fma

// ============================================================================
// Builtin declarations/implementations for FMA operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_rz(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rz, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_ru(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::ru, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_rd(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rd, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_fma_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_rz(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rz, fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_ru(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::ru, fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_rd(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rd, fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rn, fpemu_accuracy::low>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_rz(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rz, fpemu_accuracy::low>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_ru(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::ru, fpemu_accuracy::low>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_rd(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_fma<__fpemu_rounding::rd, fpemu_accuracy::low>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_fma(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_fma_unpacked<fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_fma(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_fma_unpacked<fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_fma(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_fma_unpacked<fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_fma(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_fma_unpacked<fpemu_accuracy::low>(__x, __y, __z);
}
#else
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_rz(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_ru(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_fma_rd(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_fma_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_rz(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_ru(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_fma_rd(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_rz(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_ru(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_fma_rd(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_fma(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_fma(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_fma(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_fma(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
#endif // _CCCL_FPEMU_INLINE
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_FMA_H

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_FMA_API_MERGED)
#define _CCCL_FPEMU_FMA_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// ============================================================================
// API (merged from fp64emu_fma_api.hpp)
// ============================================================================
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
fma(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_high_fma_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_fma_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_fma_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
__fma_rn(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_high_fma_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_fma_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_fma_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
__fma_rz(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_fma_rz(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_fma_rz(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_fma_rz(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_fma_rz(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
__fma_ru(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_fma_ru(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_fma_ru(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_fma_ru(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_fma_ru(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
__fma_rd(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_fma_rd(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_fma_rd(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_fma_rd(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_fma_rd(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}

template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
fma(const fpemu_unpacked<double, _Acc>& __x,
    const fpemu_unpacked<double, _Acc>& __y,
    const fpemu_unpacked<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_fma(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_fma(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_fma(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
__fma_rn(const fpemu_unpacked<double, _Acc>& __x,
         const fpemu_unpacked<double, _Acc>& __y,
         const fpemu_unpacked<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_fma(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_fma(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_fma(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
}

// Mixed-operand promoters (relocated from the class body; formerly hidden
// friends). Enabled only when at least one operand is an fpemu and at least
// one is a built-in arithmetic type: both operands are promoted to the fpemu
// type and the exact-match core above is called. Pure fpemu/fpemu calls bind
// to the cores directly; pure arithmetic calls are left to the language.

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return fma(_Fp(__x), _Fp(__y), _Fp(__z));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return __fma_rn(_Fp(__x), _Fp(__y), _Fp(__z));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> __fma_rz(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return __fma_rz(_Fp(__x), _Fp(__y), _Fp(__z));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> __fma_ru(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return __fma_ru(_Fp(__x), _Fp(__y), _Fp(__z));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> __fma_rd(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return __fma_rd(_Fp(__x), _Fp(__y), _Fp(__z));
}
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Overloads of fma for the emulated double types so the standard spelling
// cuda::std::fma selects the emulated implementation. A qualified cuda::std::fma
// call suppresses ADL, so without these it would silently narrow fpemu -> double
// (via the implicit conversion) and compute a native-double fma. These forward to
// cuda::experimental::fma, which unqualified/ADL calls already resolve to. The
// exact-type overloads cover pure fpemu/fpemu/fpemu calls (which __fpemu_mixed_v
// excludes), while the constrained overload handles mixed fpemu + arithmetic.
template <::cuda::experimental::fpemu_accuracy _Acc>
[[nodiscard]] _CCCL_API ::cuda::experimental::fpemu<double, _Acc>
fma(const ::cuda::experimental::fpemu<double, _Acc>& __x,
    const ::cuda::experimental::fpemu<double, _Acc>& __y,
    const ::cuda::experimental::fpemu<double, _Acc>& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}
template <::cuda::experimental::fpemu_accuracy _Acc>
[[nodiscard]] _CCCL_API ::cuda::experimental::fpemu_unpacked<double, _Acc>
fma(const ::cuda::experimental::fpemu_unpacked<double, _Acc>& __x,
    const ::cuda::experimental::fpemu_unpacked<double, _Acc>& __y,
    const ::cuda::experimental::fpemu_unpacked<double, _Acc>& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(::cuda::experimental::__fpemu_mixed_v<_T1, _T2, _T3>)
[[nodiscard]] _CCCL_API ::cuda::experimental::__fpemu_pick_t<_T1, _T2, _T3>
fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>
#endif // _CCCL_FPEMU_FMA_API_MERGED
