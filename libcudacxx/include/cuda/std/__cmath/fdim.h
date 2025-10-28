//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_FDIM_H
#define _CUDA_STD___CMATH_FDIM_H

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

// fdim

#if _CCCL_CHECK_BUILTIN(builtin_fdim) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FDIMF(...) __builtin_fdimf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FDIM(...)  __builtin_fdim(__VA_ARGS__)
#  define _CCCL_BUILTIN_FDIML(...) __builtin_fdiml(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fdim)

#if _CCCL_CUDA_COMPILER(CLANG) // Unresolved extern function 'fdim'
#  undef _CCCL_BUILTIN_FDIMF
#  undef _CCCL_BUILTIN_FDIM
#  undef _CCCL_BUILTIN_FDIML
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float fdim(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FDIMF)
  return _CCCL_BUILTIN_FDIMF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FDIMF ^^^ / vvv !_CCCL_BUILTIN_FDIMF vvv
  return ::fdimf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_FDIMF ^^^
}

[[nodiscard]] _CCCL_API inline float fdimf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_FDIMF)
  return _CCCL_BUILTIN_FDIMF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FDIMF ^^^ / vvv !_CCCL_BUILTIN_FDIMF vvv
  return ::fdimf(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_FDIMF ^^^
}

[[nodiscard]] _CCCL_API inline double fdim(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_FDIM)
  return _CCCL_BUILTIN_FDIM(__x, __y);
#else // ^^^ _CCCL_BUILTIN_FDIM ^^^ / vvv !_CCCL_BUILTIN_FDIM vvv
  return ::fdim(__x, __y);
#endif // ^^^ !_CCCL_BUILTIN_FDIM ^^^
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fdim(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FDIML)
  return _CCCL_BUILTIN_FDIML(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FDIML ^^^ / vvv !_CCCL_BUILTIN_FDIML vvv
  return ::fdiml(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_FDIML ^^^
}

[[nodiscard]] _CCCL_API inline long double fdiml(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_FDIML)
  return _CCCL_BUILTIN_FDIML(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_FDIML ^^^ / vvv !_CCCL_BUILTIN_FDIML vvv
  return ::fdiml(__x, __y);
#  endif // ^^^ !_CCCL_BUILTIN_FDIML ^^^
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half fdim(__half __x, __half __y) noexcept
{
  return ::__float2half(::cuda::std::fdim(::__half2float(__x), ::__half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 fdim(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return ::__float2bfloat16(::cuda::std::fdim(::__bfloat162float(__x), ::__bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr double fdim(_Tp __x, _Tp __y) noexcept
{
  return ::cuda::std::fdim(static_cast<double>(__x), static_cast<double>(__y));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_FDIM_H
