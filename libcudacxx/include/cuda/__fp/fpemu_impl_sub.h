//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_SUB_H
#define _CUDA___FP_FPEMU_IMPL_SUB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fpemu_dsub_impl.hpp
//! @brief Implementation of double-precision subtraction operations for FPEMU floating point emulation library
//!
//! This header provides the implementation of double-precision subtraction operations for the FPEMU library.
//! It includes:
//!
//! - Subtraction functions for fpemu
//! - Subtraction operators for fpemu
//! - Subtraction functions to other types
//!
//! The subtraction functions are designed to work across both host and device code
//! through appropriate decorators and provide bit-exact results matching hardware
//! floating point units.

#include <cuda/__fp/fpemu_impl.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/__fp/fpemu_impl_unpack.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Subtract two __fpbits64_unpacked
//!
//! This function subtracts two __fpbits64_unpacked.
//!
//! @param a The first __fpbits64_unpacked
//! @param b The second __fpbits64_unpacked
//! @return The result of the subtraction
template <fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64_unpacked
__internal_fp64emu_dsub_unpacked(__fpbits64_unpacked __a, __fpbits64_unpacked __b) noexcept
{
  return __internal_fp64emu_dadd_unpacked<_Acc, true>(__a, __b);
}

//! @brief Subtract two __fpbits64
//!
//! This function subtracts two __fpbits64.
//!
//! @param x The first __fpbits64
//! @param y The second __fpbits64
//! @return The result of the subtraction
template <__fpemu_rounding _Rm = __fpemu_rounding::def, fpemu_accuracy _Acc = fpemu_accuracy::def>
_CCCL_TRIVIAL_API __fpbits64 __internal_fp64emu_dsub(__fpbits64 __x, __fpbits64 __y) noexcept
{
  // Forced parameters for the subtraction operation
  constexpr fpemu_accuracy __acc_forced = fpemu_accuracy::_CCCL_FPEMU_ADD_METHOD;
  constexpr fpemu_accuracy __acc_used   = (__acc_forced != fpemu_accuracy::unset) ? __acc_forced : _Acc;

  {
    // Pass true to the dadd function to indicate that we are subtracting
    return __internal_fp64emu_dadd<_Rm, __acc_used, true>(__x, __y);
  }
} // __internal_fp64emu_dsub

// ============================================================================
// Builtin declarations/implementations for subtraction operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rz(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rz, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_ru(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::ru, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rd(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rd, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dsub_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rz(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rz, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_ru(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::ru, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rd(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rd, fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rn(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rn, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rz(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rz, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_ru(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::ru, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rd(__fpbits64 __x, __fpbits64 __y) noexcept
{
  return __internal_fp64emu_dsub<__fpemu_rounding::rd, fpemu_accuracy::low>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_dsub(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_dsub(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::high>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_dsub(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::mid>(__x, __y);
}
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_dsub(__fpbits64_unpacked __x, __fpbits64_unpacked __y) noexcept
{
  return __internal_fp64emu_dsub_unpacked<fpemu_accuracy::low>(__x, __y);
}
#else
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rz(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_ru(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_dsub_rd(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_high_dsub_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rz(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_ru(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_mid_dsub_rd(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rn(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rz(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_ru(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64 __fp64emu_low_dsub_rd(__fpbits64 x, __fpbits64 y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_dsub(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_high_dsub(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_mid_dsub(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
_CCCL_FPEMU_BUILTIN_DECL __fpbits64_unpacked
__fp64emu_unpacked_low_dsub(__fpbits64_unpacked x, __fpbits64_unpacked y) noexcept;
#endif // _CCCL_FPEMU_INLINE
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_SUB_H (builtins)

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_DSUB_API_MERGED)
#define _CCCL_FPEMU_DSUB_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// ============================================================================
// API (merged from fp64emu_dsub_api.hpp)
// ============================================================================

// Default API implementation - binary subtraction operator
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> operator-(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_high_dsub_rn(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_mid_dsub_rn(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_low_dsub_rn(__x.__bits_, __y.__bits_));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(__fp64emu_dsub_rn(__x.__bits_, __y.__bits_));
  }
} // operator-

template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dsub_rn(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_high_dsub_rn(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dsub_rn(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dsub_rn(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dsub_rz(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dsub_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dsub_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dsub_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dsub_rz(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dsub_ru(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dsub_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dsub_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dsub_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dsub_ru(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}
template <fpemu_accuracy _Acc>
_CCCL_API fpemu<double, _Acc> __dsub_rd(const fpemu<double, _Acc>& __x, const fpemu<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dsub_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_mid_dsub_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_low_dsub_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu<double, _Acc>>(
      __fp64emu_dsub_rd(__fpemu_bit_cast<__fpbits64>(__x), __fpemu_bit_cast<__fpbits64>(__y)));
  }
}

// Operator- for unpacked subtraction
template <fpemu_accuracy _Acc>
_CCCL_DEVICE_API fpemu_unpacked<double, _Acc>
operator-(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_dsub(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::mid)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_dsub(__x.__bits_, __y.__bits_));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_dsub(__x.__bits_, __y.__bits_));
  }
  else
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_dsub(__x.__bits_, __y.__bits_));
  }
} // operator-

template <fpemu_accuracy _Acc>
_CCCL_API fpemu_unpacked<double, _Acc>
__dsub_rn(const fpemu_unpacked<double, _Acc>& __x, const fpemu_unpacked<double, _Acc>& __y) noexcept
{
  if constexpr (_Acc == fpemu_accuracy::high)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_high_dsub(
      __fpemu_bit_cast<__fpbits64_unpacked>(__x), __fpemu_bit_cast<__fpbits64_unpacked>(__y)));
  }
  else if constexpr (_Acc == fpemu_accuracy::low)
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_low_dsub(
      __fpemu_bit_cast<__fpbits64_unpacked>(__x), __fpemu_bit_cast<__fpbits64_unpacked>(__y)));
  }
  else
  {
    return __fpemu_bit_cast<fpemu_unpacked<double, _Acc>>(__fp64emu_unpacked_mid_dsub(
      __fpemu_bit_cast<__fpbits64_unpacked>(__x), __fpemu_bit_cast<__fpbits64_unpacked>(__y)));
  }
}

// Mixed-operand promoters (relocated from the class body; formerly hidden
// friends). Enabled only when at least one operand is an fpemu and at least
// one is a built-in arithmetic type: both operands are promoted to the fpemu
// type and the exact-match core above is called. Pure fpemu/fpemu calls bind
// to the cores directly; pure arithmetic calls are left to the language.

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dsub_rn(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dsub_rn(_Fp(__x), _Fp(__y));
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dsub_rz(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dsub_rz(_Fp(__x), _Fp(__y));
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dsub_ru(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dsub_ru(_Fp(__x), _Fp(__y));
}

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__fpemu_mixed_v<_T1, _T2>)
_CCCL_API __fpemu_pick_t<_T1, _T2> __dsub_rd(const _T1& __x, const _T2& __y) noexcept
{
  using _Fp = __fpemu_pick_t<_T1, _T2>;
  return __dsub_rd(_Fp(__x), _Fp(__y));
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_SUB_H
