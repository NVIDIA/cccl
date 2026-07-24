//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_OTHERS_H
#define _CUDA___FP_FPEMU_IMPL_OTHERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fpemu_impl_others.h
//! @brief Implementation of MAD, DOT, and CMUL operations for FPEMU floating point emulation library
//!
//! This header provides the implementation of other operations for the FPEMU library.
//! It includes:
//!   - MAD (Multiply-Add with intermediate rounding) functions for different accuracy and range configurations
//!   - DOT (dot product) functions
//!   - CMUL (complex multiply) functions
//!   - Special case handling for NaN, inf, zero, etc
//!
//! The implementation is designed to work across both host and device code
//! through appropriate decorators and provide bit-exact results matching hardware
//! floating point units.

#define _CCCL_FP64EMU_USE_OPT_MAD_UNPACKED  1
#define _CCCL_FP64EMU_USE_OPT_DOT_UNPACKED  1
#define _CCCL_FP64EMU_USE_OPT_CMUL_UNPACKED 1

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/__fp/fpemu_impl_mul.h>
#include <cuda/__fp/fpemu_impl_sub.h>
#include <cuda/__fp/fpemu_impl_unpack.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// MAD unpacked implementation
template <fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64_unpacked
__internal_fp64emu_mad_unpacked(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_dadd_unpacked<_Acc>(__internal_fp64emu_dmul_unpacked<_Acc>(__x, __y), __z);
}

// DOT unpacked implementation
template <fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_dot_unpacked(
  __fpbits64_unpacked __x1, __fpbits64_unpacked __y1, __fpbits64_unpacked __x2, __fpbits64_unpacked __y2) noexcept
{
  return __internal_fp64emu_dadd_unpacked<_Acc>(
    __internal_fp64emu_dmul_unpacked<_Acc>(__x1, __x2), __internal_fp64emu_dmul_unpacked<_Acc>(__y1, __y2));
}

// CMPLX MUL unpacked implementation
// (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
template <fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API void __internal_fp64emu_cmul_unpacked(
  __fpbits64_unpacked __x_re,
  __fpbits64_unpacked __x_im,
  __fpbits64_unpacked __y_re,
  __fpbits64_unpacked __y_im,
  __fpbits64_unpacked& __r_re,
  __fpbits64_unpacked& __r_im) noexcept
{
  __r_re = __internal_fp64emu_dsub_unpacked<_Acc>(
    __internal_fp64emu_dmul_unpacked<_Acc>(__x_re, __y_re), __internal_fp64emu_dmul_unpacked<_Acc>(__x_im, __y_im));
  __r_im = __internal_fp64emu_dadd_unpacked<_Acc>(
    __internal_fp64emu_dmul_unpacked<_Acc>(__x_re, __y_im), __internal_fp64emu_dmul_unpacked<_Acc>(__x_im, __y_re));
  return;
}

// MAD implementation
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_mad(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::mid)
  {
    __fpbits64_unpacked __x_unpacked = __internal_fp64emu_unpack(__x);
    __fpbits64_unpacked __y_unpacked = __internal_fp64emu_unpack(__y);
    __fpbits64_unpacked __z_unpacked = __internal_fp64emu_unpack(__z);

    __fpbits64_unpacked __r_unpacked = __internal_fp64emu_mad_unpacked<_Acc>(__x_unpacked, __y_unpacked, __z_unpacked);
    return __internal_fp64emu_pack<_Rm>(__r_unpacked);
  }
  else
  {
    return __internal_fp64emu_dadd<_Rm, _Acc>(__internal_fp64emu_dmul<_Rm, _Acc>(__x, __y), __z);
  }
}

// DOT implementation
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64
__internal_fp64emu_dot(__fpbits64 __x1, __fpbits64 __y1, __fpbits64 __x2, __fpbits64 __y2) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::mid)
  {
    __fpbits64_unpacked __x1_unpacked = __internal_fp64emu_unpack(__x1);
    __fpbits64_unpacked __y1_unpacked = __internal_fp64emu_unpack(__y1);
    __fpbits64_unpacked __x2_unpacked = __internal_fp64emu_unpack(__x2);
    __fpbits64_unpacked __y2_unpacked = __internal_fp64emu_unpack(__y2);

    __fpbits64_unpacked __r_unpacked =
      __internal_fp64emu_dot_unpacked<_Acc>(__x1_unpacked, __y1_unpacked, __x2_unpacked, __y2_unpacked);
    __fpbits64 __r = __internal_fp64emu_pack<_Rm>(__r_unpacked);

    return __r;
  }
  else
  {
    __fpbits64 __r = __internal_fp64emu_dadd<_Rm, _Acc>(
      __internal_fp64emu_dmul<_Rm, _Acc>(__x1, __x2), __internal_fp64emu_dmul<_Rm, _Acc>(__y1, __y2));
    return __r;
  }
}

// CMUL implementation
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API void __internal_fp64emu_cmul(
  __fpbits64 __x_re,
  __fpbits64 __x_im,
  __fpbits64 __y_re,
  __fpbits64 __y_im,
  __fpbits64& __r_re,
  __fpbits64& __r_im) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::mid)
  {
    __fpbits64_unpacked __x_re_unpacked = __internal_fp64emu_unpack(__x_re);
    __fpbits64_unpacked __y_re_unpacked = __internal_fp64emu_unpack(__y_re);
    __fpbits64_unpacked __x_im_unpacked = __internal_fp64emu_unpack(__x_im);
    __fpbits64_unpacked __y_im_unpacked = __internal_fp64emu_unpack(__y_im);
    __fpbits64_unpacked __r_re_unpacked;
    __fpbits64_unpacked __r_im_unpacked;

    __internal_fp64emu_cmul_unpacked<_Acc>(
      __x_re_unpacked, __x_im_unpacked, __y_re_unpacked, __y_im_unpacked, __r_re_unpacked, __r_im_unpacked);

    __r_re = __internal_fp64emu_pack<_Rm>(__r_re_unpacked);
    __r_im = __internal_fp64emu_pack<_Rm>(__r_im_unpacked);

    return;
  }
  else
  {
    __fpbits64 __r_re_y_re = __internal_fp64emu_dmul<_Rm, _Acc>(__x_re, __y_re);
    __fpbits64 __r_im_y_im = __internal_fp64emu_dmul<_Rm, _Acc>(__x_im, __y_im);
    __fpbits64 __r_re_y_im = __internal_fp64emu_dmul<_Rm, _Acc>(__x_re, __y_im);
    __fpbits64 __r_im_y_re = __internal_fp64emu_dmul<_Rm, _Acc>(__x_im, __y_re);

    __r_re = __internal_fp64emu_dsub<_Rm, _Acc>(__r_re_y_re, __r_im_y_im);
    __r_im = __internal_fp64emu_dadd<_Rm, _Acc>(__r_re_y_im, __r_im_y_re);

    return;
  }
}

_CCCL_TRIVIAL_API __fpbits64_unpacked __internal_fp64emu_neg_unpacked(__fpbits64_unpacked __x) noexcept
{
  __x.sign = __invert_msb(__x.sign);
  return __x;
}

_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_neg(__fpbits64 __x) noexcept
{
  __uint32x2 __t = ::cuda::std::bit_cast<__uint32x2>(__x);
  __t.x[1]       = __invert_msb(__t.x[1]);
  __x            = ::cuda::std::bit_cast<uint64_t>(__t);
  return __x;
}

// ============================================================================
// Builtin declarations/implementations for MAD, DOT, CMUL, NEG operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)

// mad (packed)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mad_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_mad<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_mad_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_mad<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_mad_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_mad<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_mad_rn(__fpbits64 __x, __fpbits64 __y, __fpbits64 __z) noexcept
{
  return __internal_fp64emu_mad<__fpemu_rounding::rn, fpemu_accuracy::low>(__x, __y, __z);
}

// dot (packed)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_dot_rn(__fpbits64 __x1, __fpbits64 __y1, __fpbits64 __x2, __fpbits64 __y2) noexcept
{
  return __internal_fp64emu_dot<__fpemu_rounding::rn, fpemu_accuracy::high>(__x1, __y1, __x2, __y2);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_high_dot_rn(__fpbits64 __x1, __fpbits64 __y1, __fpbits64 __x2, __fpbits64 __y2) noexcept
{
  return __internal_fp64emu_dot<__fpemu_rounding::rn, fpemu_accuracy::high>(__x1, __y1, __x2, __y2);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_mid_dot_rn(__fpbits64 __x1, __fpbits64 __y1, __fpbits64 __x2, __fpbits64 __y2) noexcept
{
  return __internal_fp64emu_dot<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x1, __y1, __x2, __y2);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_low_dot_rn(__fpbits64 __x1, __fpbits64 __y1, __fpbits64 __x2, __fpbits64 __y2) noexcept
{
  return __internal_fp64emu_dot<__fpemu_rounding::rn, fpemu_accuracy::low>(__x1, __y1, __x2, __y2);
}

// cmul (packed)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_cmul_rn(
  __fpbits64 __x_re,
  __fpbits64 __x_im,
  __fpbits64 __y_re,
  __fpbits64 __y_im,
  __fpbits64& __r_re,
  __fpbits64& __r_im) noexcept
{
  __internal_fp64emu_cmul<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_high_cmul_rn(
  __fpbits64 __x_re,
  __fpbits64 __x_im,
  __fpbits64 __y_re,
  __fpbits64 __y_im,
  __fpbits64& __r_re,
  __fpbits64& __r_im) noexcept
{
  __internal_fp64emu_cmul<__fpemu_rounding::rn, fpemu_accuracy::high>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_mid_cmul_rn(
  __fpbits64 __x_re,
  __fpbits64 __x_im,
  __fpbits64 __y_re,
  __fpbits64 __y_im,
  __fpbits64& __r_re,
  __fpbits64& __r_im) noexcept
{
  __internal_fp64emu_cmul<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_low_cmul_rn(
  __fpbits64 __x_re,
  __fpbits64 __x_im,
  __fpbits64 __y_re,
  __fpbits64 __y_im,
  __fpbits64& __r_re,
  __fpbits64& __r_im) noexcept
{
  __internal_fp64emu_cmul<__fpemu_rounding::rn, fpemu_accuracy::low>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}

// neg (packed)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_neg(__fpbits64 __x) noexcept
{
  return __internal_fp64emu_neg(__x);
}

// mad (unpacked)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mad(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_mad_unpacked<fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_mad(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_mad_unpacked<fpemu_accuracy::high>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_mad(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_mad_unpacked<fpemu_accuracy::mid>(__x, __y, __z);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_mad(__fpbits64_unpacked __x, __fpbits64_unpacked __y, __fpbits64_unpacked __z) noexcept
{
  return __internal_fp64emu_mad_unpacked<fpemu_accuracy::low>(__x, __y, __z);
}

// dot (unpacked)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_dot(
  __fpbits64_unpacked __x1, __fpbits64_unpacked __y1, __fpbits64_unpacked __x2, __fpbits64_unpacked __y2) noexcept
{
  return __internal_fp64emu_dot_unpacked<fpemu_accuracy::mid>(__x1, __y1, __x2, __y2);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_dot(
  __fpbits64_unpacked __x1, __fpbits64_unpacked __y1, __fpbits64_unpacked __x2, __fpbits64_unpacked __y2) noexcept
{
  return __internal_fp64emu_dot_unpacked<fpemu_accuracy::high>(__x1, __y1, __x2, __y2);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_dot(
  __fpbits64_unpacked __x1, __fpbits64_unpacked __y1, __fpbits64_unpacked __x2, __fpbits64_unpacked __y2) noexcept
{
  return __internal_fp64emu_dot_unpacked<fpemu_accuracy::mid>(__x1, __y1, __x2, __y2);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_dot(
  __fpbits64_unpacked __x1, __fpbits64_unpacked __y1, __fpbits64_unpacked __x2, __fpbits64_unpacked __y2) noexcept
{
  return __internal_fp64emu_dot_unpacked<fpemu_accuracy::low>(__x1, __y1, __x2, __y2);
}

// cmul (unpacked)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_cmul(
  __fpbits64_unpacked __x_re,
  __fpbits64_unpacked __x_im,
  __fpbits64_unpacked __y_re,
  __fpbits64_unpacked __y_im,
  __fpbits64_unpacked& __r_re,
  __fpbits64_unpacked& __r_im) noexcept
{
  __internal_fp64emu_cmul_unpacked<fpemu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_high_cmul(
  __fpbits64_unpacked __x_re,
  __fpbits64_unpacked __x_im,
  __fpbits64_unpacked __y_re,
  __fpbits64_unpacked __y_im,
  __fpbits64_unpacked& __r_re,
  __fpbits64_unpacked& __r_im) noexcept
{
  __internal_fp64emu_cmul_unpacked<fpemu_accuracy::high>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_mid_cmul(
  __fpbits64_unpacked __x_re,
  __fpbits64_unpacked __x_im,
  __fpbits64_unpacked __y_re,
  __fpbits64_unpacked __y_im,
  __fpbits64_unpacked& __r_re,
  __fpbits64_unpacked& __r_im) noexcept
{
  __internal_fp64emu_cmul_unpacked<fpemu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_low_cmul(
  __fpbits64_unpacked __x_re,
  __fpbits64_unpacked __x_im,
  __fpbits64_unpacked __y_re,
  __fpbits64_unpacked __y_im,
  __fpbits64_unpacked& __r_re,
  __fpbits64_unpacked& __r_im) noexcept
{
  __internal_fp64emu_cmul_unpacked<fpemu_accuracy::low>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im);
}

// neg (unpacked)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_neg(__fpbits64_unpacked __x) noexcept
{
  return __internal_fp64emu_neg_unpacked(__x);
}

#else // LTO mode - declarations only

// mad (packed)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mad_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_mad_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_mad_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_mad_rn(__fpbits64 x, __fpbits64 y, __fpbits64 z) noexcept;

// dot (packed)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_dot_rn(__fpbits64 x1, __fpbits64 y1, __fpbits64 x2, __fpbits64 y2) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_high_dot_rn(__fpbits64 x1, __fpbits64 y1, __fpbits64 x2, __fpbits64 y2) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_mid_dot_rn(__fpbits64 x1, __fpbits64 y1, __fpbits64 x2, __fpbits64 y2) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64
__fp64emu_low_dot_rn(__fpbits64 x1, __fpbits64 y1, __fpbits64 x2, __fpbits64 y2) noexcept;

// cmul (packed)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_cmul_rn(
  __fpbits64 x_re, __fpbits64 x_im, __fpbits64 y_re, __fpbits64 y_im, __fpbits64& r_re, __fpbits64& r_im) noexcept;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_high_cmul_rn(
  __fpbits64 x_re, __fpbits64 x_im, __fpbits64 y_re, __fpbits64 y_im, __fpbits64& r_re, __fpbits64& r_im) noexcept;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_mid_cmul_rn(
  __fpbits64 x_re, __fpbits64 x_im, __fpbits64 y_re, __fpbits64 y_im, __fpbits64& r_re, __fpbits64& r_im) noexcept;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_low_cmul_rn(
  __fpbits64 x_re, __fpbits64 x_im, __fpbits64 y_re, __fpbits64 y_im, __fpbits64& r_re, __fpbits64& r_im) noexcept;

// neg (packed)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_neg(__fpbits64 x) noexcept;

// mad (unpacked)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mad(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_mad(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_mad(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_mad(__fpbits64_unpacked x, __fpbits64_unpacked y, __fpbits64_unpacked z) noexcept;

// dot (unpacked)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_dot(
  __fpbits64_unpacked x1, __fpbits64_unpacked y1, __fpbits64_unpacked x2, __fpbits64_unpacked y2) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_high_dot(
  __fpbits64_unpacked x1, __fpbits64_unpacked y1, __fpbits64_unpacked x2, __fpbits64_unpacked y2) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_mid_dot(
  __fpbits64_unpacked x1, __fpbits64_unpacked y1, __fpbits64_unpacked x2, __fpbits64_unpacked y2) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_low_dot(
  __fpbits64_unpacked x1, __fpbits64_unpacked y1, __fpbits64_unpacked x2, __fpbits64_unpacked y2) noexcept;

// cmul (unpacked)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_cmul(
  __fpbits64_unpacked x_re,
  __fpbits64_unpacked x_im,
  __fpbits64_unpacked y_re,
  __fpbits64_unpacked y_im,
  __fpbits64_unpacked& r_re,
  __fpbits64_unpacked& r_im) noexcept;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_high_cmul(
  __fpbits64_unpacked x_re,
  __fpbits64_unpacked x_im,
  __fpbits64_unpacked y_re,
  __fpbits64_unpacked y_im,
  __fpbits64_unpacked& r_re,
  __fpbits64_unpacked& r_im) noexcept;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_mid_cmul(
  __fpbits64_unpacked x_re,
  __fpbits64_unpacked x_im,
  __fpbits64_unpacked y_re,
  __fpbits64_unpacked y_im,
  __fpbits64_unpacked& r_re,
  __fpbits64_unpacked& r_im) noexcept;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_low_cmul(
  __fpbits64_unpacked x_re,
  __fpbits64_unpacked x_im,
  __fpbits64_unpacked y_re,
  __fpbits64_unpacked y_im,
  __fpbits64_unpacked& r_re,
  __fpbits64_unpacked& r_im) noexcept;

// neg (unpacked)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked __fp64emu_unpacked_neg(__fpbits64_unpacked x) noexcept;

#endif // _CCCL_FPEMU_INLINE
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_OTHERS_H

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_OTHERS_API_MERGED)
#define _CCCL_FPEMU_OTHERS_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// ============================================================================
// API (merged from fp64emu_others_api.hpp)
// ============================================================================

// Unary negation operator - member function implementation
template <typename _FpType, fpemu_accuracy _Acc>
_CCCL_API fpemu<_FpType, _Acc> fpemu<_FpType, _Acc>::operator-() const noexcept
{
  fpemu __temp(*this);
  __temp.__bits_ = __fp64emu_neg(__temp.__bits_);
  return __temp;
}

template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
mad(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_high_mad_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_mad_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_mad_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
__mad_rn(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y, const fpemu<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_high_mad_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_mad_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_mad_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x),
      ::cuda::std::bit_cast<__fpbits64>(__y),
      ::cuda::std::bit_cast<__fpbits64>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc>
dot(const fpemu<double, _Acc>& __x1,
    const fpemu<double, _Acc>& __y1,
    const fpemu<double, _Acc>& __x2,
    const fpemu<double, _Acc>& __y2) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_high_dot_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x1),
      ::cuda::std::bit_cast<__fpbits64>(__y1),
      ::cuda::std::bit_cast<__fpbits64>(__x2),
      ::cuda::std::bit_cast<__fpbits64>(__y2)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_low_dot_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x1),
      ::cuda::std::bit_cast<__fpbits64>(__y1),
      ::cuda::std::bit_cast<__fpbits64>(__x2),
      ::cuda::std::bit_cast<__fpbits64>(__y2)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_dot_rn(
      ::cuda::std::bit_cast<__fpbits64>(__x1),
      ::cuda::std::bit_cast<__fpbits64>(__y1),
      ::cuda::std::bit_cast<__fpbits64>(__x2),
      ::cuda::std::bit_cast<__fpbits64>(__y2)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API void
cmul(const fpemu<double, _Acc>& __x_re,
     const fpemu<double, _Acc>& __x_im,
     const fpemu<double, _Acc>& __y_re,
     const fpemu<double, _Acc>& __y_im,
     fpemu<double, _Acc>& __r_re,
     fpemu<double, _Acc>& __r_im) noexcept
{
  const __fpbits64 __xr = ::cuda::std::bit_cast<__fpbits64>(__x_re);
  const __fpbits64 __xi = ::cuda::std::bit_cast<__fpbits64>(__x_im);
  const __fpbits64 __yr = ::cuda::std::bit_cast<__fpbits64>(__y_re);
  const __fpbits64 __yi = ::cuda::std::bit_cast<__fpbits64>(__y_im);
  // The builtins write their results through non-const references, so compute into
  // local raw-bits temporaries and construct the outputs from them (bits is private).
  __fpbits64 __rr{};
  __fpbits64 __ri{};
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    __fp64emu_high_cmul_rn(__xr, __xi, __yr, __yi, __rr, __ri);
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    __fp64emu_low_cmul_rn(__xr, __xi, __yr, __yi, __rr, __ri);
  }
  else
  {
    __fp64emu_mid_cmul_rn(__xr, __xi, __yr, __yi, __rr, __ri);
  }
  __r_re = ::cuda::std::bit_cast<fpemu<double, _Acc>>(__rr);
  __r_im = ::cuda::std::bit_cast<fpemu<double, _Acc>>(__ri);
}

// Unary negation operator for unpacked - member function implementation
template <typename _FpType, fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<_FpType, _Acc> fpemu_unpacked<_FpType, _Acc>::operator-() const noexcept
{
  fpemu_unpacked __temp(*this);
  __temp.__bits_ = __fp64emu_unpacked_neg(__temp.__bits_);
  return __temp;
}

template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
mad(const fpemu_unpacked<double, _Acc>& __x,
    const fpemu_unpacked<double, _Acc>& __y,
    const fpemu_unpacked<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_mad(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_mad(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_mad(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
__mad_rn(const fpemu_unpacked<double, _Acc>& __x,
         const fpemu_unpacked<double, _Acc>& __y,
         const fpemu_unpacked<double, _Acc>& __z) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_mad(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_mad(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_mad(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__z)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
dot(const fpemu_unpacked<double, _Acc>& __x1,
    const fpemu_unpacked<double, _Acc>& __y1,
    const fpemu_unpacked<double, _Acc>& __x2,
    const fpemu_unpacked<double, _Acc>& __y2) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_dot(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x1),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y1),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x2),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y2)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_dot(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x1),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y1),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x2),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y2)));
  }
  else
  {
    return ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_dot(
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x1),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y1),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__x2),
      ::cuda::std::bit_cast<__fpbits64_unpacked>(__y2)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API void
cmul(const fpemu_unpacked<double, _Acc>& __x_re,
     const fpemu_unpacked<double, _Acc>& __x_im,
     const fpemu_unpacked<double, _Acc>& __y_re,
     const fpemu_unpacked<double, _Acc>& __y_im,
     fpemu_unpacked<double, _Acc>& __r_re,
     fpemu_unpacked<double, _Acc>& __r_im) noexcept
{
  const __fpbits64_unpacked __xr = ::cuda::std::bit_cast<__fpbits64_unpacked>(__x_re);
  const __fpbits64_unpacked __xi = ::cuda::std::bit_cast<__fpbits64_unpacked>(__x_im);
  const __fpbits64_unpacked __yr = ::cuda::std::bit_cast<__fpbits64_unpacked>(__y_re);
  const __fpbits64_unpacked __yi = ::cuda::std::bit_cast<__fpbits64_unpacked>(__y_im);
  // The builtins write their results through non-const references, so compute into
  // local raw-bits temporaries and construct the outputs from them (bits is private).
  __fpbits64_unpacked __rr{};
  __fpbits64_unpacked __ri{};
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    __fp64emu_unpacked_high_cmul(__xr, __xi, __yr, __yi, __rr, __ri);
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    __fp64emu_unpacked_low_cmul(__xr, __xi, __yr, __yi, __rr, __ri);
  }
  else
  {
    __fp64emu_unpacked_mid_cmul(__xr, __xi, __yr, __yi, __rr, __ri);
  }
  __r_re = ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__rr);
  __r_im = ::cuda::std::bit_cast<fpemu_unpacked<double, _Acc>>(__ri);
}

// Mixed-operand promoters (relocated from the class body; formerly hidden
// friends). Enabled only when at least one operand is an fpemu and at least
// one is a built-in arithmetic type: both operands are promoted to the fpemu
// type and the exact-match core above is called. Pure fpemu/fpemu calls bind
// to the cores directly; pure arithmetic calls are left to the language.

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return mad(_Fp(__x), _Fp(__y), _Fp(__z));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3> __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3>;
  return __mad_rn(_Fp(__x), _Fp(__y), _Fp(__z));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3, class _T4)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3, _T4>)
_CCCL_API __fpemu_pick_t<_T1, _T2, _T3, _T4>
dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3, _T4>;
  return dot(_Fp(__x1), _Fp(__y1), _Fp(__x2), _Fp(__y2));
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3, class _T4)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2, _T3, _T4>)
_CCCL_API void
cmul(const _T1& __x_re,
     const _T2& __x_im,
     const _T3& __y_re,
     const _T4& __y_im,
     __fpemu_pick_t<_T1, _T2, _T3, _T4>& __r_re,
     __fpemu_pick_t<_T1, _T2, _T3, _T4>& __r_im) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2, _T3, _T4>;
  cmul(_Fp(__x_re), _Fp(__x_im), _Fp(__y_re), _Fp(__y_im), __r_re, __r_im);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CCCL_FPEMU_OTHERS_API_MERGED
