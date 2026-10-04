//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_CMP_H
#define _CUDA___FP_FPMP_IMPL_CMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_impl_cmp.h - fpmp2 comparison operations
    ==================================================================================================
    Per-operation implementation core split out of <cuda/__fp/fpmp_impl.h>. It carries the
    comparison operations
    for the fpmp2 double-word type, for both the header-only (inline) mode and the library
    (_CCCL_FPMP_USE_LIB) mode. All shared macros, the fp128 vocabulary type, and the __fpmp_*
    error-free-transform primitives live in <cuda/__fp/fpmp_impl.h>, which this header includes.
*/

#include <cuda/__fp/fpmp_impl.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)
/*
 * --------------------------------------------------------------------
 * Comparison operations
 * --------------------------------------------------------------------
 */
// == comparison
template <typename _FpType>
_CCCL_FPMP_CORE_API bool
__fpmp2_cmp_eq(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo) noexcept
{
  return __x_hi == __y_hi && __x_lo == __y_lo;
}

// != comparison
template <typename _FpType>
_CCCL_FPMP_CORE_API bool
__fpmp2_cmp_ne(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo) noexcept
{
  return __x_hi != __y_hi || __x_lo != __y_lo;
}

// < comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
template <typename _FpType>
_CCCL_FPMP_CORE_API bool
__fpmp2_cmp_lt(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo) noexcept
{
  return __x_hi < __y_hi || (__x_hi == __y_hi && __x_lo < __y_lo);
}

// > comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
template <typename _FpType>
_CCCL_FPMP_CORE_API bool
__fpmp2_cmp_gt(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo) noexcept
{
  return __x_hi > __y_hi || (__x_hi == __y_hi && __x_lo > __y_lo);
}

// <= comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
template <typename _FpType>
_CCCL_FPMP_CORE_API bool
__fpmp2_cmp_le(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo) noexcept
{
  return __x_hi < __y_hi || (__x_hi == __y_hi && __x_lo <= __y_lo);
}

// >= comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
template <typename _FpType>
_CCCL_FPMP_CORE_API bool
__fpmp2_cmp_ge(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo) noexcept
{
  return __x_hi > __y_hi || (__x_hi == __y_hi && __x_lo >= __y_lo);
}

#else // _CCCL_FPMP_USE_LIB

// -- fp32 (single precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL bool
__fp32mp2_cmp_eq(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp32mp2_cmp_ne(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp32mp2_cmp_lt(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp32mp2_cmp_gt(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp32mp2_cmp_le(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp32mp2_cmp_ge(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;

// -- fp64 (double precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL bool
__fp64mp2_cmp_eq(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp64mp2_cmp_ne(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp64mp2_cmp_lt(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp64mp2_cmp_gt(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp64mp2_cmp_le(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool
__fp64mp2_cmp_ge(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;

// -- type-generic template declarations (dispatch to fp32/fp64) --
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_eq(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_ne(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_lt(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_gt(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_le(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_ge(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;

// -- fp32 template specializations --
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_eq<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept
{
  return __fp32mp2_cmp_eq(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_ne<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept
{
  return __fp32mp2_cmp_ne(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_lt<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept
{
  return __fp32mp2_cmp_lt(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_gt<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept
{
  return __fp32mp2_cmp_gt(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_le<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept
{
  return __fp32mp2_cmp_le(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_ge<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept
{
  return __fp32mp2_cmp_ge(__x_hi, __x_lo, __y_hi, __y_lo);
}

// -- fp64 template specializations --
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_eq<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept
{
  return __fp64mp2_cmp_eq(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_ne<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept
{
  return __fp64mp2_cmp_ne(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_lt<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept
{
  return __fp64mp2_cmp_lt(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_gt<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept
{
  return __fp64mp2_cmp_gt(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_le<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept
{
  return __fp64mp2_cmp_le(__x_hi, __x_lo, __y_hi, __y_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline bool
__fpmp2_cmp_ge<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept
{
  return __fp64mp2_cmp_ge(__x_hi, __x_lo, __y_hi, __y_lo);
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_CMP_H
