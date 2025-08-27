//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___COMPLEX_ROOTS_H
#define _CUDA_STD___COMPLEX_ROOTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/copysign.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/math.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Specialization of 1/sqrt(x) to use rsqrt on device when available.
// Curiously, for its use here in csqrt, since the device fp32 rsqrtf
// maps directly to device asm, while the fp64 rsqrt maps to an llvm-ir
// intrinsic, with the compiler optimizations 1/sqrt(x) generates faster
// code than rsqrt(x) for fp64. While rqsrtf is better for fp32.
// So only specialize for fp32.
template <class _Tp>
[[nodiscard]] _CCCL_API inline _Tp __internal_rsqrt(_Tp __x) noexcept
{
  if constexpr (is_same_v<_Tp, float>)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::rsqrtf(__x);))
  }
  return _Tp(1) / ::cuda::std::sqrt(__x);
}

// sqrt

template <class _Tp>
[[nodiscard]] _CCCL_API inline complex<_Tp> sqrt(const complex<_Tp>& __x) noexcept
{
  using __uint_t = __fp_storage_of_t<_Tp>;

  constexpr int32_t __max_exponent = __fp_exp_max_v<__fp_format_of_v<_Tp>>;
  constexpr int32_t __mant_nbits   = __fp_mant_nbits_v<__fp_format_of_v<_Tp>>;
  constexpr int32_t __exp_bias     = __fp_exp_bias_v<__fp_format_of_v<_Tp>>;

  const _Tp __re = __x.real();
  const _Tp __im = __x.imag();

  if (::cuda::std::isinf(__im))
  {
    return complex<_Tp>{numeric_limits<_Tp>::infinity(), __im};
  }

  if ((__re == _Tp(0)) && (__im == _Tp(0)))
  {
    return complex<_Tp>{_Tp(0), __im};
  }

  // pre-check to see if we over/underflow:
  _Tp __x_abs_sq = ::cuda::std::fma(__re, __re, __im * __im);

  // Get some bounds where __re +- |__x| won't overflow.
  // Doesn't need to be too exact, enough to cover extremal cases.
  // overflow bound = sqrt(MAX_FLOAT / 2)
  // underflow bound similar, but tweaked to allow for normalizing denormal calculation.
  constexpr __uint_t __overflow_bound_exp =
    (static_cast<__uint_t>((static_cast<__uint_t>(__max_exponent - 1) >> 1) + __exp_bias) << __mant_nbits)
    | __fp_explicit_bit_mask_of_v<_Tp>;
  constexpr __uint_t __underflow_bound_exp =
    (static_cast<__uint_t>((static_cast<__uint_t>(-__max_exponent + __mant_nbits) >> 1) + __exp_bias) << __mant_nbits)
    | __fp_explicit_bit_mask_of_v<_Tp>;

  _Tp __overflow_bound  = ::cuda::std::__fp_from_storage<_Tp>(__overflow_bound_exp);
  _Tp __underflow_bound = ::cuda::std::__fp_from_storage<_Tp>(__underflow_bound_exp);

  // Prepare some range-reduction that simplifies the calculation and avoids needing the full hypot function.
  // Alas there is not a single splitting point that works for all values, so we split into 3 intervals.
  // (denormals are to blame)
  _Tp __ldexp_factor_1 = _Tp(2.0); // Power of 2 of the form 2^(2m + 1)
  _Tp __ldexp_factor_2 = _Tp(0.5); // 1/sqrt(2*ldexp_factor_1)
  _Tp __ldexp_combined = _Tp(1.0); // __ldexp_factor_1*__ldexp_factor_2

  if (__x_abs_sq > __overflow_bound)
  {
    // Construct some compile-time __ldexp_factor_* powers of 2 for range reduction.
    // To guarantee |__re|, |__im| is <= sqrt(MAX_FLOAT / 2)
    // we can multiply by some C with C <= 1.0 / sqrt(MAX_FLOAT / 2)
    // We also want C to be a power of 2, and for our algorithm, an odd power of 2.

    // Divide ~MAX_FLOAT by 2, sqrt.
    // We also overshoot a little (by + 2), to allow error accumulation near the boundary.
    // To not push us into intermediate INF territory.
    constexpr int __reduction_exponent = ((__max_exponent - 1) >> 1) + 2;

    // Make sure it's odd, and not bigger.
    constexpr int __reduced_exponent = __reduction_exponent - (1 - (__reduction_exponent & 0x1));

    // Negate, add bias, and shift into the exponent mask.
    constexpr __uint_t __lxexp_1_uint =
      (__uint_t(__exp_bias - __reduced_exponent) << __uint_t(__mant_nbits)) | __fp_explicit_bit_mask_of_v<_Tp>;

    // __ldexp_factor_2 = 1/sqrt(2*ldexp_factor_1)
    constexpr __uint_t __lxexp_2_uint =
      ((__exp_bias + (__uint_t(__reduced_exponent - 1) >> 1)) << __uint_t(__mant_nbits))
      | __fp_explicit_bit_mask_of_v<_Tp>;

    // Set our scaling values:
    __ldexp_factor_1 = ::cuda::std::__fp_from_storage<_Tp>(__lxexp_1_uint);
    __ldexp_factor_2 = ::cuda::std::__fp_from_storage<_Tp>(__lxexp_2_uint);
    __ldexp_combined = __ldexp_factor_1 * __ldexp_factor_2;
  }

  if (__x_abs_sq < __underflow_bound)
  {
    // Need some extra range compared to the overflow version to account for denormals.
    // Similar otherwise.
    constexpr int32_t __min_denom_exponent = -__max_exponent - __mant_nbits;

    constexpr int __reduction_exponent = ((__min_denom_exponent - 1) >> 1) - 2;
    constexpr int __reduced_exponent   = __reduction_exponent + (1 - (__reduction_exponent & 0x1));

    constexpr __uint_t __lxexp_1_uint =
      (__uint_t(__exp_bias - __reduced_exponent) << __uint_t(__mant_nbits)) | __fp_explicit_bit_mask_of_v<_Tp>;
    constexpr __uint_t __lxexp_2_uint =
      ((__exp_bias + (__uint_t(__reduced_exponent) >> 1)) << __uint_t(__mant_nbits)) | __fp_explicit_bit_mask_of_v<_Tp>;

    // Set our scaling values:
    __ldexp_factor_1 = ::cuda::std::__fp_from_storage<_Tp>(__lxexp_1_uint);
    __ldexp_factor_2 = ::cuda::std::__fp_from_storage<_Tp>(__lxexp_2_uint);
    __ldexp_combined = __ldexp_factor_1 * __ldexp_factor_2;
  }

  const _Tp __re_scaled = __re * __ldexp_factor_1;
  const _Tp __im_scaled = __im * __ldexp_factor_1;

  // An inlined hypot.
  // Surprisingly, the final accuracy gets worse if you try and make this hypot calculation more accurate
  // for the case (__im >> __re) by swapping the fma inputs, as the final result is not symmetrical in __im/__re.
  _Tp __x_abs_scaled = ::cuda::std::sqrt(::cuda::std::fma(__re_scaled, __re_scaled, __im_scaled * __im_scaled));

  // Add in the hypot inf-nan override here to avoid a complicated inf/nan check at the start.
  if (::cuda::std::isinf(__im) || ::cuda::std::isinf(__re))
  {
    __x_abs_scaled = numeric_limits<_Tp>::infinity();
  }

  // These would be the ideal terms we want to compute, however we can get catastrophic
  // cancellation in either (depending on if __re_scaled > 0 or not).
  const _Tp __ans_re_sq = __x_abs_scaled + __re_scaled;
  const _Tp __ans_im_sq = __x_abs_scaled - __re_scaled;

  // Get rid of catastrophic cancellation issues by using conjugate:
  //  for x > 0:
  //  x - sqrt(x^2 + y^2) = -y^2 / (x + sqrt(x^2 + y^2))
  //
  // Similarly for x < 0:
  //        x + sqrt(x^2 + y^2) = -y^2/(x - sqrt(x^2 + y^2))

  const bool __im_part_is_hard = (__re_scaled > _Tp(0.0));
  _Tp __easy_part              = __im_part_is_hard ? __ans_re_sq : __ans_im_sq;

  _Tp __sqrt_easy_part = ::cuda::std::__internal_rsqrt<_Tp>(__easy_part);

  // (__im_scaled) can have under/overflowed when scaled so we go back to using (__im) briefly.
  // Multiply by (__im) last to avoid over/underflow:
  const _Tp __hard_part = (__im) * (__ldexp_combined * __sqrt_easy_part);

  __easy_part = __ldexp_factor_2 * ::cuda::std::sqrt(__easy_part);

  // Don't need fabs on the second line, but better for code generation:
  const _Tp __ans_re = __im_part_is_hard ? __easy_part : ::cuda::std::fabs(__hard_part);
  const _Tp __ans_im = __im_part_is_hard ? ::cuda::std::fabs(__hard_part) : __easy_part;

  return complex<_Tp>{__ans_re, ::cuda::std::copysign(__ans_im, __im)};
}

#if _LIBCUDACXX_HAS_NVBF16()
template <>
_CCCL_API inline complex<__nv_bfloat16> sqrt(const complex<__nv_bfloat16>& __x) noexcept
{
  return complex<__nv_bfloat16>{::cuda::std::sqrt(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _LIBCUDACXX_HAS_NVFP16()
template <>
_CCCL_API inline complex<__half> sqrt(const complex<__half>& __x) noexcept
{
  return complex<__half>{::cuda::std::sqrt(complex<float>{__x})};
}
#endif // _LIBCUDACXX_HAS_NVFP16()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___COMPLEX_ROOTS_H
