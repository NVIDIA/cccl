//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_ERROR_FUNCTIONS_H
#define _CUDA_STD___CMATH_ERROR_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_integral.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// erf

#if _CCCL_CHECK_BUILTIN(builtin_erf) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ERFF(...) __builtin_erff(__VA_ARGS__)
#  define _CCCL_BUILTIN_ERF(...)  __builtin_erf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ERFL(...) __builtin_erfl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_erf)

#if _CCCL_CUDA_COMPILER(CLANG) // Unresolved extern function 'erf'
#  undef _CCCL_BUILTIN_ERFF
#  undef _CCCL_BUILTIN_ERF
#  undef _CCCL_BUILTIN_ERFL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float erf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ERFF)
  return _CCCL_BUILTIN_ERFF(__x);
#else // ^^^ _CCCL_BUILTIN_ERFF ^^^ / vvv !_CCCL_BUILTIN_ERFF vvv
  return ::erff(__x);
#endif // ^^^ !_CCCL_BUILTIN_ERFF ^^^
}

[[nodiscard]] _CCCL_API inline float erff(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ERFF)
  return _CCCL_BUILTIN_ERFF(__x);
#else // ^^^ _CCCL_BUILTIN_ERFF ^^^ / vvv !_CCCL_BUILTIN_ERFF vvv
  return ::erff(__x);
#endif // ^^^ !_CCCL_BUILTIN_ERFF ^^^
}

[[nodiscard]] _CCCL_API inline double erf(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ERF)
  return _CCCL_BUILTIN_ERF(__x);
#else // ^^^ _CCCL_BUILTIN_ERF ^^^ / vvv !_CCCL_BUILTIN_ERF vvv
  return ::erf(__x);
#endif // ^^^ !_CCCL_BUILTIN_ERF ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double erf(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ERFL)
  return _CCCL_BUILTIN_ERFL(__x);
#  else // ^^^ _CCCL_BUILTIN_ERFL ^^^ / vvv !_CCCL_BUILTIN_ERFL vvv
  return ::erfl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_ERFL ^^^
}

[[nodiscard]] _CCCL_API inline long double erfl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ERFL)
  return _CCCL_BUILTIN_ERFL(__x);
#  else // ^^^ _CCCL_BUILTIN_ERFL ^^^ / vvv !_CCCL_BUILTIN_ERFL vvv
  return ::erfl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_ERFL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half erf(__half __x) noexcept
{
  return ::__float2half(::cuda::std::erf(::__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 erf(__nv_bfloat16 __x) noexcept
{
  return ::__float2bfloat16(::cuda::std::erf(::__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API inline double erf(_Tp __x) noexcept
{
  return ::cuda::std::erf(static_cast<double>(__x));
}

// erfc

#if _CCCL_CHECK_BUILTIN(builtin_ercf) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ERFCF(...) __builtin_erfcf(__VA_ARGS__)
#  define _CCCL_BUILTIN_ERFC(...)  __builtin_erfc(__VA_ARGS__)
#  define _CCCL_BUILTIN_ERFCL(...) __builtin_erfcl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ercf)

#if _CCCL_CUDA_COMPILER(CLANG) // Unresolved extern function 'erfc'
#  undef _CCCL_BUILTIN_ERFCF
#  undef _CCCL_BUILTIN_ERFC
#  undef _CCCL_BUILTIN_ERFCL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float erfc(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ERFCF)
  return _CCCL_BUILTIN_ERFCF(__x);
#else // ^^^ _CCCL_BUILTIN_ERFCF ^^^ / vvv !_CCCL_BUILTIN_ERFCF vvv
  return ::erfcf(__x);
#endif // ^^^ !_CCCL_BUILTIN_ERFCF ^^^
}

[[nodiscard]] _CCCL_API inline float erfcf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ERFCF)
  return _CCCL_BUILTIN_ERFCF(__x);
#else // ^^^ _CCCL_BUILTIN_ERFCF ^^^ / vvv !_CCCL_BUILTIN_ERFCF vvv
  return ::erfcf(__x);
#endif // ^^^ !_CCCL_BUILTIN_ERFCF ^^^
}

[[nodiscard]] _CCCL_API inline double erfc(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ERFC)
  return _CCCL_BUILTIN_ERFC(__x);
#else // ^^^ _CCCL_BUILTIN_ERFC ^^^ / vvv !_CCCL_BUILTIN_ERFC vvv
  return ::erfc(__x);
#endif // ^^^ !_CCCL_BUILTIN_ERFC ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double erfc(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ERFCL)
  return _CCCL_BUILTIN_ERFCL(__x);
#  else // ^^^ _CCCL_BUILTIN_ERFCL ^^^ / vvv !_CCCL_BUILTIN_ERFCL vvv
  return ::erfcl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_ERFCL ^^^
}

[[nodiscard]] _CCCL_API inline long double erfcl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ERFCL)
  return _CCCL_BUILTIN_ERFCL(__x);
#  else // ^^^ _CCCL_BUILTIN_ERFCL ^^^ / vvv !_CCCL_BUILTIN_ERFCL vvv
  return ::erfcl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_ERFCL ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half erfc(__half __x) noexcept
{
  return ::__float2half(::cuda::std::erfc(::__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 erfc(__nv_bfloat16 __x) noexcept
{
  return ::__float2bfloat16(::cuda::std::erfc(::__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API inline double erfc(_Tp __x) noexcept
{
  return ::cuda::std::erfc(static_cast<double>(__x));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_ERROR_FUNCTIONS_H
