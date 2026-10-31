// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_ROOTS_H
#define _CUDA_STD___CMATH_ROOTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/cuda_fp_types.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__host_stdlib/math.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// sqrt

#if _CCCL_CHECK_BUILTIN(builtin_sqrt) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SQRTF(...) __builtin_sqrtf(__VA_ARGS__)
#  define _CCCL_BUILTIN_SQRT(...)  __builtin_sqrt(__VA_ARGS__)
#  define _CCCL_BUILTIN_SQRTL(...) __builtin_sqrtl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_sqrt)

#if _CCCL_HAS_FLOAT128() && _CCCL_CHECK_BUILTIN(builtin_sqrtf128)
#  define _CCCL_BUILTIN_SQRTF128(...) __builtin_sqrtf128(__VA_ARGS__)
#endif // _CCCL_HAS_FLOAT128() && _CCCL_CHECK_BUILTIN(builtin_sqrtf128)

// nvcc doesn't implement __builtin_sqrtf128 on device
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_BUILTIN_SQRTF128
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

#if _CCCL_HAS_FLOAT128()
// Portable, correctly rounded square root of __float128, used when neither a host builtin nor a device intrinsic is
// available
[[nodiscard]] _CCCL_API inline __float128 __sqrt_fp128_impl(__float128 __x) noexcept
{
  using __storage_t          = __fp_storage_of_t<__float128>;
  constexpr auto __fmt       = __fp_format_of_v<__float128>;
  constexpr int __mant_nbits = __fp_mant_nbits_v<__fmt>;
  constexpr int __exp_bias   = __fp_exp_bias_v<__fmt>;
  const auto __bits          = ::cuda::std::__fp_get_storage(__x);
  const auto __abs_bits      = static_cast<__storage_t>(__bits & __fp_exp_mant_mask_of_v<__float128>);

  if (__abs_bits > __fp_exp_mask_of_v<__float128>)
  {
    // quiet the NaN
    return __x + __x;
  }
  if (__abs_bits == 0)
  {
    // sqrt(-0) is -0
    return __x;
  }
  if (__bits != __abs_bits)
  {
    // sqrt(x < 0) is a domain error, return NaN and raise FE_INVALID
    // NOLINTNEXTLINE(misc-redundant-expression): intentional, raises FE_INVALID
    return (__x - __x) / (__x - __x);
  }
  if (__abs_bits == __fp_exp_mask_of_v<__float128>)
  {
    // +inf
    return __x;
  }

  // x = __mant * 2^(__exp - __mant_nbits) with the implicit bit set in __mant
  auto __mant = static_cast<__storage_t>(__bits & __fp_mant_mask_of_v<__float128>);
  int __exp   = ::cuda::std::__fp_get_exp(__x);
  if ((__bits & __fp_exp_mask_of_v<__float128>) == 0)
  {
    __exp = 1 - __exp_bias;
    while ((__mant >> __mant_nbits) == 0)
    {
      __mant <<= 1;
      --__exp;
    }
  }
  else
  {
    __mant |= __storage_t{1} << __mant_nbits;
  }
  if (__exp & 1)
  {
    __mant <<= 1;
    --__exp;
  }

  // Digit-by-digit square root of __mant * 2^__mant_nbits. __rem is the remainder scaled down by the current bit, so
  // it stays well within the 128-bit storage.
  __storage_t __root       = 0;
  __storage_t __twice_root = 0;
  __storage_t __rem        = __mant;
  for (auto __bit = __storage_t{1} << __mant_nbits; __bit != 0; __bit >>= 1)
  {
    const __storage_t __t = __twice_root + __bit;
    if (__t <= __rem)
    {
      __rem -= __t;
      __twice_root = __t + __bit;
      __root += __bit;
    }
    __rem <<= 1;
  }
  // Adding __root, which still carries the implicit bit, increments the exponent field by one
  const auto __exp_field = static_cast<__storage_t>(__exp / 2 + __exp_bias - 1);
  const auto __result =
    ::cuda::std::__fp_from_storage<__float128>(static_cast<__storage_t>((__exp_field << __mant_nbits) + __root));
  if (__rem == 0)
  {
    return __result;
  }

  // The exact root is never halfway between two representable values. Adding a quarter or three quarters of an ulp
  // lets the floating-point addition round in the current rounding mode and raise FE_INEXACT. This also handles a
  // rounding carry into the exponent.
  const auto __delta_bits =
    (__rem > __twice_root)
      ? ((__exp_field - __mant_nbits) << __mant_nbits) | (__storage_t{1} << (__mant_nbits - 1))
      : (__exp_field - __mant_nbits - 1) << __mant_nbits;
  return __result + ::cuda::std::__fp_from_storage<__float128>(__delta_bits);
}
#endif // _CCCL_HAS_FLOAT128()

[[nodiscard]] _CCCL_API inline float sqrt(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SQRTF)
  return _CCCL_BUILTIN_SQRTF(__x);
#else // ^^^ _CCCL_BUILTIN_SQRTF ^^^ // vvv !_CCCL_BUILTIN_SQRTF vvv
  return ::sqrtf(__x);
#endif // !_CCCL_BUILTIN_SQRTF
}

[[nodiscard]] _CCCL_API inline float sqrtf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SQRTF)
  return _CCCL_BUILTIN_SQRTF(__x);
#else // ^^^ _CCCL_BUILTIN_SQRTF ^^^ // vvv !_CCCL_BUILTIN_SQRTF vvv
  return ::sqrtf(__x);
#endif // !_CCCL_BUILTIN_SQRTF
}

[[nodiscard]] _CCCL_API inline double sqrt(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SQRT)
  return _CCCL_BUILTIN_SQRT(__x);
#else // ^^^ _CCCL_BUILTIN_SQRT ^^^ // vvv !_CCCL_BUILTIN_SQRT vvv
  return ::sqrt(__x);
#endif // !_CCCL_BUILTIN_SQRT
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double sqrt(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SQRTL)
  return _CCCL_BUILTIN_SQRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_SQRTL ^^^ // vvv !_CCCL_BUILTIN_SQRTL vvv
  return ::sqrtl(__x);
#  endif // !_CCCL_BUILTIN_SQRTL
}

[[nodiscard]] _CCCL_API inline long double sqrtl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SQRTL)
  return _CCCL_BUILTIN_SQRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_SQRTL ^^^ // vvv !_CCCL_BUILTIN_SQRTL vvv
  return ::sqrtl(__x);
#  endif // !_CCCL_BUILTIN_SQRTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half sqrt(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsqrt(__x);), (return __float2half(::cuda::std::sqrt(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 sqrt(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hsqrt(__x);), (return __float2bfloat16(::cuda::std::sqrt(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _CCCL_HAS_FLOAT128()
[[nodiscard]] _CCCL_API inline __float128 sqrt(__float128 __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SQRTF128)
  return _CCCL_BUILTIN_SQRTF128(__x);
#  else // ^^^ _CCCL_BUILTIN_SQRTF128 ^^^ / vvv !_CCCL_BUILTIN_SQRTF128 vvv
  NV_IF_TARGET(NV_PROVIDES_SM_100, (return ::__nv_fp128_sqrt(__x);))
  return ::cuda::std::__sqrt_fp128_impl(__x);
#  endif // ^^^ !_CCCL_BUILTIN_SQRTF128 ^^^
}
#endif // _CCCL_HAS_FLOAT128()

template <class _Integer, enable_if_t<is_integral_v<_Integer>, int> = 0>
[[nodiscard]] _CCCL_API inline double sqrt(_Integer __x) noexcept
{
  return ::cuda::std::sqrt((double) __x);
}

// cbrt

#if _CCCL_CHECK_BUILTIN(builtin_cbrt) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_CBRTF(...) __builtin_cbrtf(__VA_ARGS__)
#  define _CCCL_BUILTIN_CBRT(...)  __builtin_cbrt(__VA_ARGS__)
#  define _CCCL_BUILTIN_CBRTL(...) __builtin_cbrtl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_cbrt)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "cbrt"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_CBRTF
#  undef _CCCL_BUILTIN_CBRT
#  undef _CCCL_BUILTIN_CBRTL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float cbrt(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CBRTF)
  return _CCCL_BUILTIN_CBRTF(__x);
#else // ^^^ _CCCL_BUILTIN_CBRTF ^^^ // vvv !_CCCL_BUILTIN_CBRTF vvv
  return ::cbrtf(__x);
#endif // !_CCCL_BUILTIN_CBRTF
}

[[nodiscard]] _CCCL_API inline float cbrtf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_CBRTF)
  return _CCCL_BUILTIN_CBRTF(__x);
#else // ^^^ _CCCL_BUILTIN_CBRTF ^^^ // vvv !_CCCL_BUILTIN_CBRTF vvv
  return ::cbrtf(__x);
#endif // !_CCCL_BUILTIN_CBRTF
}

[[nodiscard]] _CCCL_API inline double cbrt(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_CBRT)
  return _CCCL_BUILTIN_CBRT(__x);
#else // ^^^ _CCCL_BUILTIN_CBRT ^^^ // vvv !_CCCL_BUILTIN_CBRT vvv
  return ::cbrt(__x);
#endif // !_CCCL_BUILTIN_CBRT
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double cbrt(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CBRTL)
  return _CCCL_BUILTIN_CBRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_CBRTL ^^^ // vvv !_CCCL_BUILTIN_CBRTL vvv
  return ::cbrtl(__x);
#  endif // !_CCCL_BUILTIN_CBRTL
}

[[nodiscard]] _CCCL_API inline long double cbrtl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_CBRTL)
  return _CCCL_BUILTIN_CBRTL(__x);
#  else // ^^^ _CCCL_BUILTIN_CBRTL ^^^ // vvv !_CCCL_BUILTIN_CBRTL vvv
  return ::cbrtl(__x);
#  endif // !_CCCL_BUILTIN_CBRTL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half cbrt(__half __x) noexcept
{
  return __float2half(::cuda::std::cbrt(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 cbrt(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(::cuda::std::cbrt(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<is_integral_v<_Integer>, int> = 0>
[[nodiscard]] _CCCL_API inline double cbrt(_Integer __x) noexcept
{
  return ::cuda::std::cbrt((double) __x);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_ROOTS_H
