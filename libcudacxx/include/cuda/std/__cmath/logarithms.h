// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_LOGARITHMS_H
#define _CUDA_STD___CMATH_LOGARITHMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/fpclassify.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// log

#if _CCCL_CHECK_BUILTIN(builtin_log) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOGF(...) __builtin_logf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG(...)  __builtin_log(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOGL(...) __builtin_logl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "logf"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOGF
#  undef _CCCL_BUILTIN_LOG
#  undef _CCCL_BUILTIN_LOGL
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float log(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGF)
  return _CCCL_BUILTIN_LOGF(__x);
#else // ^^^ _CCCL_BUILTIN_LOGF ^^^ / vvv !_CCCL_BUILTIN_LOGF vvv
  return ::logf(__x);
#endif // !_CCCL_BUILTIN_LOGF
}

[[nodiscard]] _CCCL_API inline float logf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGF)
  return _CCCL_BUILTIN_LOGF(__x);
#else // ^^^ _CCCL_BUILTIN_LOGF ^^^ / vvv !_CCCL_BUILTIN_LOGF vvv
  return ::logf(__x);
#endif // !_CCCL_BUILTIN_LOGF
}

[[nodiscard]] _CCCL_API inline double log(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG)
  return _CCCL_BUILTIN_LOG(__x);
#else // ^^^ _CCCL_BUILTIN_LOG ^^^ / vvv !_CCCL_BUILTIN_LOG vvv
  return ::log(__x);
#endif // !_CCCL_BUILTIN_LOG
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double log(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOGL)
  return _CCCL_BUILTIN_LOGL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOGL ^^^ / vvv !_CCCL_BUILTIN_LOGL vvv
  return ::logl(__x);
#  endif // !_CCCL_BUILTIN_LOGL
}

[[nodiscard]] _CCCL_API inline long double logl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOGL)
  return _CCCL_BUILTIN_LOGL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOGL ^^^ / vvv !_CCCL_BUILTIN_LOGL vvv
  return ::logl(__x);
#  endif // !_CCCL_BUILTIN_LOGL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half log(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (return ::hlog(__x);), ({
                      float __vf            = __half2float(__x);
                      __vf                  = ::cuda::std::logf(__vf);
                      __half_raw __ret_repr = ::__float2half_rn(__vf);

                      ::cuda::std::uint16_t __repr = ::cuda::std::__fp_get_storage(__x);
                      switch (__repr)
                      {
                        case 7544:
                          __ret_repr.x -= 1;
                          break;

                        default:;
                      }

                      return __ret_repr;
                    }))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 log(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hlog(__x);), (return __float2bfloat16(::cuda::std::logf(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<is_integral_v<_Integer>, int> = 0>
[[nodiscard]] _CCCL_API inline double log(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG)
  return _CCCL_BUILTIN_LOG((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG ^^^ / vvv !_CCCL_BUILTIN_LOG vvv
  return ::log((double) __x);
#endif // !_CCCL_BUILTIN_LOG
}

// log10

#if _CCCL_CHECK_BUILTIN(builtin_log10) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOG10F(...) __builtin_log10f(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG10(...)  __builtin_log10(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG10L(...) __builtin_log10l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log10)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "log10f"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOG10F
#  undef _CCCL_BUILTIN_LOG10
#  undef _CCCL_BUILTIN_LOG10L
#endif //  _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float log10(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10F)
  return _CCCL_BUILTIN_LOG10F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG10F ^^^ / vvv !_CCCL_BUILTIN_LOG10F vvv
  return ::log10f(__x);
#endif // !_CCCL_BUILTIN_LOG10F
}

[[nodiscard]] _CCCL_API inline float log10f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10F)
  return _CCCL_BUILTIN_LOG10F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG10F ^^^ / vvv !_CCCL_BUILTIN_LOG10F vvv
  return ::log10f(__x);
#endif // !_CCCL_BUILTIN_LOG10F
}

[[nodiscard]] _CCCL_API inline double log10(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10)
  return _CCCL_BUILTIN_LOG10(__x);
#else // ^^^ _CCCL_BUILTIN_LOG10 ^^^ / vvv !_CCCL_BUILTIN_LOG10 vvv
  return ::log10(__x);
#endif // !_CCCL_BUILTIN_LOG10
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double log10(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG10L)
  return _CCCL_BUILTIN_LOG10L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG10L ^^^ / vvv !_CCCL_BUILTIN_LOG10L vvv
  return ::log10l(__x);
#  endif // !_CCCL_BUILTIN_LOG10L
}

[[nodiscard]] _CCCL_API inline long double log10l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG10L)
  return _CCCL_BUILTIN_LOG10L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG10L ^^^ / vvv !_CCCL_BUILTIN_LOG10L vvv
  return ::log10l(__x);
#  endif // !_CCCL_BUILTIN_LOG10L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half log10(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_53, (return ::hlog10(__x);), (return __float2half(::cuda::std::log10f(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 log10(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hlog10(__x);), (return __float2bfloat16(::cuda::std::log10f(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<is_integral_v<_Integer>, int> = 0>
[[nodiscard]] _CCCL_API inline double log10(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10)
  return _CCCL_BUILTIN_LOG10((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG10 ^^^ / vvv !_CCCL_BUILTIN_LOG10 vvv
  return ::log10((double) __x);
#endif // !_CCCL_BUILTIN_LOG10
}

// ilogb

template <class _Tp>
[[nodiscard]] _CCCL_API inline constexpr int __ilogb_impl(_Tp __x) noexcept
{
  switch (::cuda::std::fpclassify(__x))
  {
    case FP_ZERO:
      return FP_ILOGB0;
    case FP_NAN:
      return FP_ILOGBNAN;
    case FP_INFINITE:
      return numeric_limits<int>::max();
  }

  constexpr auto __fmt = __fp_format_of_v<_Tp>;
  const int __exp      = ::cuda::std::__fp_get_exp(__x);
  if (__exp > __fp_exp_max_v<__fmt>)
  {
    return numeric_limits<int>::max();
  }
  else if (__exp < __fp_exp_min_v<__fmt>)
  {
    return numeric_limits<int>::min();
  }
  else
  {
    return __exp;
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_API inline constexpr int ilogb(_Tp __x) noexcept
{
  if constexpr (is_integral_v<_Tp>)
  {
    return ::cuda::std::__ilogb_impl(static_cast<double>(__x));
  }
  else
  {
    return ::cuda::std::__ilogb_impl(__x);
  }
}

[[nodiscard]] _CCCL_API inline constexpr int ilogbf(float __x) noexcept
{
  return ::cuda::std::__ilogb_impl(__x);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline constexpr int ilogbl(long double __x) noexcept
{
  return ::cuda::std::__ilogb_impl(__x);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

// log1p

#if _CCCL_CHECK_BUILTIN(builtin_log1p) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOG1PF(...) __builtin_log1pf(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG1P(...)  __builtin_log1p(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG1PL(...) __builtin_log1pl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log1p)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "log1p"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOG1PF
#  undef _CCCL_BUILTIN_LOG1P
#  undef _CCCL_BUILTIN_LOG1PL
#endif //  _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float log1p(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1PF)
  return _CCCL_BUILTIN_LOG1PF(__x);
#else // ^^^ _CCCL_BUILTIN_LOG1PF ^^^ / vvv !_CCCL_BUILTIN_LOG1PF vvv
  return ::log1pf(__x);
#endif // !_CCCL_BUILTIN_LOG1PF
}

[[nodiscard]] _CCCL_API inline float log1pf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1PF)
  return _CCCL_BUILTIN_LOG1PF(__x);
#else // ^^^ _CCCL_BUILTIN_LOG1PF ^^^ / vvv !_CCCL_BUILTIN_LOG1PF vvv
  return ::log1pf(__x);
#endif // !_CCCL_BUILTIN_LOG1PF
}

[[nodiscard]] _CCCL_API inline double log1p(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1P)
  return _CCCL_BUILTIN_LOG1P(__x);
#else // ^^^ _CCCL_BUILTIN_LOG1P ^^^ / vvv !_CCCL_BUILTIN_LOG1P vvv
  return ::log1p(__x);
#endif // !_CCCL_BUILTIN_LOG1P
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double log1p(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG1PL)
  return _CCCL_BUILTIN_LOG1PL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG1PL ^^^ / vvv !_CCCL_BUILTIN_LOG1PL vvv
  return ::log1pl(__x);
#  endif // !_CCCL_BUILTIN_LOG1PL
}

[[nodiscard]] _CCCL_API inline long double log1pl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG1PL)
  return _CCCL_BUILTIN_LOG1PL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG1PL ^^^ / vvv !_CCCL_BUILTIN_LOG1PL vvv
  return ::log1pl(__x);
#  endif // !_CCCL_BUILTIN_LOG1PL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half log1p(__half __x) noexcept
{
  return __float2half(::cuda::std::log1pf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 log1p(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(::cuda::std::log1pf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<is_integral_v<_Integer>, int> = 0>
[[nodiscard]] _CCCL_API inline double log1p(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1P)
  return _CCCL_BUILTIN_LOG1P((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG1P ^^^ / vvv !_CCCL_BUILTIN_LOG1P vvv
  return ::log1p((double) __x);
#endif // !_CCCL_BUILTIN_LOG1P
}

// log2

#if _CCCL_CHECK_BUILTIN(builtin_log2) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_LOG2F(...) __builtin_log2f(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG2(...)  __builtin_log2(__VA_ARGS__)
#  define _CCCL_BUILTIN_LOG2L(...) __builtin_log2l(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_log1)

// clang-cuda fails with fatal error: error in backend: Undefined external symbol "log2f"
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_LOG2F
#  undef _CCCL_BUILTIN_LOG2
#  undef _CCCL_BUILTIN_LOG2L
#endif // _CCCL_CUDA_COMPILER(CLANG)

[[nodiscard]] _CCCL_API inline float log2(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2F)
  return _CCCL_BUILTIN_LOG2F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG2F ^^^ / vvv !_CCCL_BUILTIN_LOG2F vvv
  return ::log2f(__x);
#endif // !_CCCL_BUILTIN_LOG2F
}

[[nodiscard]] _CCCL_API inline float log2f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2F)
  return _CCCL_BUILTIN_LOG2F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG2F ^^^ / vvv !_CCCL_BUILTIN_LOG2F vvv
  return ::log2f(__x);
#endif // !_CCCL_BUILTIN_LOG2F
}

[[nodiscard]] _CCCL_API inline double log2(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2)
  return _CCCL_BUILTIN_LOG2(__x);
#else // ^^^ _CCCL_BUILTIN_LOG2 ^^^ / vvv !_CCCL_BUILTIN_LOG2 vvv
  return ::log2(__x);
#endif // !_CCCL_BUILTIN_LOG2
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double log2(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG2L)
  return _CCCL_BUILTIN_LOG2L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG2L ^^^ / vvv !_CCCL_BUILTIN_LOG2L vvv
  return ::log2l(__x);
#  endif // !_CCCL_BUILTIN_LOG2L
}

[[nodiscard]] _CCCL_API inline long double log2l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG2L)
  return _CCCL_BUILTIN_LOG2L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG2L ^^^ / vvv !_CCCL_BUILTIN_LOG2L vvv
  return ::log2l(__x);
#  endif // !_CCCL_BUILTIN_LOG2L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
[[nodiscard]] _CCCL_API inline __half log2(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_53, (return ::hlog2(__x);), (return __float2half(::cuda::std::log2f(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
[[nodiscard]] _CCCL_API inline __nv_bfloat16 log2(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hlog2(__x);), (return __float2bfloat16(::cuda::std::log2f(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<is_integral_v<_Integer>, int> = 0>
[[nodiscard]] _CCCL_API inline double log2(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2)
  return _CCCL_BUILTIN_LOG2((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG2 ^^^ / vvv !_CCCL_BUILTIN_LOG2 vvv
  return ::log2((double) __x);
#endif // !_CCCL_BUILTIN_LOG2
}

// logb

template <class _Tp>
[[nodiscard]] _CCCL_API inline constexpr _Tp __logb_impl(_Tp __x) noexcept
{
  switch (::cuda::std::fpclassify(__x))
  {
    case FP_ZERO:
      return ::cuda::std::__fp_neg<_Tp>(::cuda::std::__fp_inf<_Tp>());
    case FP_NAN:
      return ::cuda::std::__fp_nan<_Tp>();
    case FP_INFINITE:
      return ::cuda::std::__fp_inf<_Tp>();
    default:
      break;
  }
#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  return static_cast<_Tp>(::cuda::std::__fp_get_exp(__x));
#else // ^^^ _CCCL_HAS_CONSTEXPR_BIT_CAST() ^^^ / vvv !_CCCL_HAS_CONSTEXPR_BIT_CAST() vvv
  // We need to go through the slow emulation for old GCC
  if constexpr (__fp_is_native_type_v<_Tp>)
  {
    __x                      = ::cuda::std::fabs(__x);
    unsigned long long __exp = 0;
    while (__x >= _Tp(numeric_limits<_Tp>::radix))
    {
      __x /= numeric_limits<_Tp>::radix;
      __exp += 1;
    }
    return static_cast<_Tp>(__exp);
  }
  else
  {
    return static_cast<_Tp>(::cuda::std::__fp_get_exp(__x));
  }
#endif // !_CCCL_HAS_CONSTEXPR_BIT_CAST()
}

[[nodiscard]] _CCCL_API inline constexpr float logbf(float __x) noexcept
{
  return ::cuda::std::__logb_impl(__x);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline constexpr long double logbl(long double __x) noexcept
{
  return ::cuda::std::__logb_impl(__x);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

template <class _Tp, enable_if_t<__is_extended_arithmetic_v<_Tp>, int> = 0>
[[nodiscard]] _CCCL_API inline constexpr conditional_t<is_integral_v<_Tp>, double, _Tp> logb(_Tp __x) noexcept
{
  if constexpr (is_integral_v<_Tp>)
  {
    return ::cuda::std::__logb_impl(static_cast<double>(__x));
  }
  else
  {
    return ::cuda::std::__logb_impl(__x);
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_LOGARITHMS_H
