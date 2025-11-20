//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_LOGARITHMS_H
#define _CUDA_STD___COMPLEX_LOGARITHMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/arg.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/math.h>
#include <cuda/std/__complex/nvbf16.h>
#include <cuda/std/__complex/nvfp16.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/numbers>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// log

// 0.5 * log1p on [-0.25, 0.5]:
[[nodiscard]] _CCCL_API inline float __internal_unsafe_log1p_poly(float __x) noexcept
{
  float __log1p_poly = 0.5f * -4.50736098e-2f;
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * 0.10464530f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * -0.13162985f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * 0.14478821f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * -0.16647165f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * 0.19990806f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * -0.25000098f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * 0.33333451f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * -0.5f);
  __log1p_poly       = ::cuda::std::fmaf(__log1p_poly, __x, 0.5f * 1.0f);
  __log1p_poly *= __x;

  return __log1p_poly;
}

// 0.5 * log1p on [-0.25, 0.5]:
[[nodiscard]] _CCCL_API inline double __internal_unsafe_log1p_poly(double __x) noexcept
{
  double __log1p_poly = 0.5 * -7.09111630733153322503e-3;
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 2.66022308034025677103e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -4.72224506011362787916e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 5.65972221291394378406e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -5.77824005018098968423e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 5.89181707234033569254e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -6.22470356657013820789e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 6.66071144648464480431e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -7.143929943708668406366e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 7.692821470788910320771e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -8.33332949182035986890e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 9.09088791817835584208e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -9.99999883706452485921e-2);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 0.11111111583539913517);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -0.12500000038299266536);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 0.14285714280067426940);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -0.16666666666197160751);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 0.20000000000032358560);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -0.25000000000001904032);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 0.33333333333333270421);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * -0.5);
  __log1p_poly        = ::cuda::std::fma(__log1p_poly, __x, 0.5 * 1.0);
  __log1p_poly *= __x;

  return __log1p_poly;
}

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> log(const complex<_Tp>& __x)
{
  // Uint of the same size as our fp type.
  // Shouldn't need make_unsigned, but just in case:
  using __uint_t = make_unsigned_t<__fp_storage_of_t<_Tp>>;

  // Some needed constants:
  constexpr __uint_t __mant_mask = __fp_mant_mask_of_v<_Tp>;
  constexpr __uint_t __exp_mask  = __fp_exp_mask_of_v<_Tp>;

  constexpr int32_t __mant_nbits = __fp_mant_nbits_v<__fp_format_of_v<_Tp>>;
  constexpr int32_t __exp_bias   = __fp_exp_bias_v<__fp_format_of_v<_Tp>>;

  // Cut off the hi and low bit of __exp_mask.
  // 0x3F000000 for fp32,
  // 0x3FE0000000000000 for fp64 etc
  constexpr __uint_t __exp_mask_of_half = ((__exp_mask >> (__mant_nbits + 2)) << (__mant_nbits + 1));

  const _Tp __real_abs = ::cuda::std::fabs(__x.real());
  const _Tp __imag_abs = ::cuda::std::fabs(__x.imag());

  _Tp __max = (__real_abs > __imag_abs) ? __real_abs : __imag_abs;
  _Tp __min = (__real_abs > __imag_abs) ? __imag_abs : __real_abs;

  // We would like to range reduce these values so that abs(x) ~ 1, as we'll take the log of this.
  // The below code inlines and removes these two calls:
  //    _Tp __max_reduced = ::cuda::std::frexpf(__max, &__exp);
  //    _Tp __min_reduced = ::cuda::std::ldexpf(__min, -__exp);

  // Unbiased exponent of 2.0*__max
  int32_t __exp = static_cast<int32_t>(::cuda::std::bit_cast<__uint_t>(__max) >> __mant_nbits) - __exp_bias + 1;

  // Quick frexp:
  __uint_t __max_reduced_as_uint = (::cuda::std::bit_cast<__uint_t>(__max) & __mant_mask) | __exp_mask_of_half;
  _Tp __max_reduced              = ::cuda::std::bit_cast<_Tp>(__max_reduced_as_uint);

  // Create an exponent for an inline ldexp(__min, -__exp)
  __uint_t __exp_neg_as_uint = (static_cast<__uint_t>(__exp_bias - __exp) << __mant_nbits);
  _Tp __exp_neg              = ::cuda::std::bit_cast<_Tp>(__exp_neg_as_uint);

  _Tp __min_reduced = __min * __exp_neg;

  // Slowpath, denormal/nan/zero/rare-underflow:
  if ((__exp == (1 - __exp_bias)) || (__exp >= __exp_bias))
  {
    // Here __exp can also be (eg for double) 3073, as fabs(NaN) can still return a "negative" NaN,
    // And our exponent extraction does not strip the sign. We can still check for this with
    // "__exp >= 1025" as opposed to "__exp == 1025", which only catches "positive" NaNs.
    if (__max == _Tp(0.0) || __exp >= (__exp_bias + 2))
    {
      // NaN/inf/0.0 (inf doesn't matter, gets fixed later by hypot)
      __max_reduced = __max;
    }
    else
    {
      if (__exp >= __exp_bias)
      { // Here __exp is (eg for double) only 1023 or 1024.
        // Create a fast ldexp power of 2 as above underflows,
        // and split it into two separate multiplications.
        // Inlined version of this code:
        //   __min_reduced = ::cuda::std::ldexp(__min, -__exp);
        __uint_t __ldexp_factor_2_uint = (static_cast<__uint_t>(__exp_bias + __mant_nbits - __exp) << __mant_nbits);
        __uint_t __two_m_mant_bits     = static_cast<__uint_t>(-__mant_nbits + __exp_bias) << __mant_nbits;

        _Tp __ldexp_factor_1 = ::cuda::std::bit_cast<_Tp>(__two_m_mant_bits); // 2^(-__mant_nbits)
        _Tp __ldexp_factor_2 = ::cuda::std::bit_cast<_Tp>(__ldexp_factor_2_uint);
        __min_reduced        = (__min * __ldexp_factor_1) * __ldexp_factor_2;
      }
      else
      {
        // __max is denormal (so __min is also denormal or 0.0)
        // Scale things up by 2^__mant_nbits then do the fast ldexp.
        __uint_t __two_mant_bits = static_cast<__uint_t>(__mant_nbits + __exp_bias) << __mant_nbits;
        _Tp __ldexp_factor       = ::cuda::std::bit_cast<_Tp>(__two_mant_bits);
        __max_reduced            = __max * __ldexp_factor; // 2^__mant_nbits
        __min_reduced            = __min * __ldexp_factor; // 2^__mant_nbits;

        int32_t __exp_no_denorm_bias =
          static_cast<int32_t>(::cuda::std::bit_cast<__uint_t>(__max_reduced) >> __mant_nbits) - __exp_bias + 1;
        __uint_t ldexp_factor_no_denorm_bias = static_cast<__uint_t>(__exp_bias - __exp_no_denorm_bias) << __mant_nbits;

        __max_reduced *= ::cuda::std::bit_cast<_Tp>(ldexp_factor_no_denorm_bias);
        __min_reduced *= ::cuda::std::bit_cast<_Tp>(ldexp_factor_no_denorm_bias);

        __exp = __exp_no_denorm_bias - __mant_nbits;
      }
    }
  }

  // We now have __max and __min reduced so that 0.5 <= __max <= 1.0.
  // However, we need to have it so hypot(__max, __min)^2 is reduced.
  // At the moment we have:
  //   0.5 <= hypot(__min_reduced, __max_reduced) <= sqrt(2)
  // We will take the logarithm of this, for the most accuracy (and to reduce log1p polynomial length),
  // we would like to make sure that __hypot_sq_scaled is close to 1.
  _Tp __hypot_sq_scaled = ::cuda::std::fma(__max_reduced, __max_reduced, __min_reduced * __min_reduced);

  if (__hypot_sq_scaled < _Tp(0.5))
  {
    __max_reduced *= _Tp(2.0);
    __min_reduced *= _Tp(2.0);
    __exp -= 1;
  }

  // hypot(__max_reduced, __min_reduced) is now ~1.0, and we now want log(hypot())).
  // We can end up with large ulp errors due to catastrophic cancellation with hypot, however.
  // To prevent this we instead calculate:
  //        ((real^2 +  imag^2) - 1)
  // accurately and use log1p.
  _Tp max_2_hi = __max_reduced * __max_reduced;
  _Tp max_2_lo = ::cuda::std::fma(__max_reduced, __max_reduced, -max_2_hi);

  _Tp min_2_hi = __min_reduced * __min_reduced;
  _Tp min_2_lo = ::cuda::std::fma(__min_reduced, __min_reduced, -min_2_hi);

  _Tp sum_hi = max_2_hi + min_2_hi;
  _Tp sum_lo = min_2_hi + (max_2_hi - sum_hi);

  // Exact where it matters, with the previous range reduction.
  sum_hi -= _Tp(1.0);

  // Compiler can rearrange the sum in brackets, same max error. Need all terms.
  __hypot_sq_scaled = sum_hi + (sum_lo + max_2_lo + min_2_lo);

  // We now need log1p(__hypot_sq_scaled).
  // The range of __hypot_sq_scaled is too large for a simple polynomial at the moment,
  // and the log1p function inself is quite heavy. We do yet more reduction using:
  //    log1p(x) = -ln(C) + log1p(C-1 + C*x)

  _Tp __exp_d = static_cast<_Tp>(__exp);

  // C = 2: log1p = ln(1/2) + log1p(1 + 2*x)
  if (__hypot_sq_scaled < _Tp(-0.25))
  {
    __hypot_sq_scaled = ::cuda::std::fma(_Tp(2.0), __hypot_sq_scaled, _Tp(1.0));
    __exp_d -= _Tp(0.5);
  }

  // C = 0.5: log1p = ln(2) + log1p(0.5 + 0.5*x)
  if (__hypot_sq_scaled >= _Tp(0.5))
  {
    __hypot_sq_scaled = ::cuda::std::fma(_Tp(0.5), __hypot_sq_scaled, _Tp(-0.5));
    __exp_d += _Tp(0.5);
  }

  // __hypot_sq_scaled is now in [-0.25, 0.5], we can use a log1p polynomial estimate.
  _Tp __log1p_poly = __internal_unsafe_log1p_poly(__hypot_sq_scaled);

  // Scale our answer back up.
  _Tp __abs_rescaled = ::cuda::std::fma(__numbers<_Tp>::__ln2(), __exp_d, __log1p_poly);

  // Fix x == 0.0
  if (__x.real() == _Tp(0.0) && __x.imag() == _Tp(0.0))
  {
    __abs_rescaled = -numeric_limits<_Tp>::infinity();
  }

  // Fix hypot inf/nan case:
  if ((__max == numeric_limits<_Tp>::infinity()) || (__min == numeric_limits<_Tp>::infinity()))
  {
    __abs_rescaled = numeric_limits<_Tp>::infinity();
  }

  return complex<_Tp>(__abs_rescaled, ::cuda::std::atan2(__x.imag(), __x.real()));
}

#if _LIBCUDACXX_HAS_NVBF16()
template <>
// The general template for log when generated for __nv_bfloat16 is both worse for
// accuracy and slower than the fp32 version.
_CCCL_API inline complex<__nv_bfloat16> log(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{::cuda::std::log(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
// The general template for log when generated for __half is both worse for
// accuracy and slower than the fp32 version.
_CCCL_API inline complex<__half> log(const complex<__half>& __x)
{
  return complex<__half>{::cuda::std::log(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

// log10

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> log10(const complex<_Tp>& __x)
{
  return ::cuda::std::log(__x) * __numbers<_Tp>::__log10e();
}

#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> log10(const complex<__nv_bfloat16>& __x)
{
  return complex<__nv_bfloat16>{::cuda::std::log10(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> log10(const complex<__half>& __x)
{
  return complex<__half>{::cuda::std::log10(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___COMPLEX_LOGARITHMS_H
