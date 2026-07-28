//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_LIMITS_H
#define _CUDA___FP_FPMP_LIMITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp_limits.h - cuda::std::numeric_limits specialization for fpmp2
    ======================================================================================================
    Provides a cuda::std::numeric_limits<> specialization for the multi-precision double-word types
    fpmp2<FpType, met> (fp32mp2 = double-float, fp64mp2 = double-double), mirroring the standard
    std::numeric_limits interface.

    Conventions (see docs/libcudacxx/fp/fpmp_spec.rst):
    -------------------------------------------------------------------------
    A value is stored as a non-overlapping pair (hi, lo) of the underlying IEEE-754 FpType. The
    reported characteristics are therefore derived from numeric_limits<FpType>:

    - digits (mantissa bits) = 2 * digits(FpType) - 2
        fp32mp2 -> 2*24 - 2 =  46 bits   (epsilon = 2^-45)
        fp64mp2 -> 2*53 - 2 = 104 bits   (epsilon = 2^-103)
      The "-2" reflects the guaranteed contiguous precision of a normalized double-word (the two
      halves must stay non-overlapping), matching the library's published mantissa bit counts.

    - Exponent range follows the double-double model: the maximum exponent matches FpType (the hi
      component caps at FpType's max), while the minimum normalized exponent is raised by digits(FpType)
      so that BOTH halves remain normal (min() is the smallest all-normal double-word).

    - The format is not IEEE-754 (is_iec559 = false), but it inherits Inf / NaN / subnormal support and
      round-to-nearest behavior from the underlying FpType arithmetic.
*/

#include <cuda/__fp/fpmp.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// constexpr 2^__e for FpType (host/device, no <cmath>). __e is small in magnitude here and every
// result used by the numeric_limits specialization is an exact power of two, so the repeated product
// is exact.
template <class _FpType>
[[nodiscard]] _CCCL_API constexpr _FpType __fpmp_limits_exp2(int __e) noexcept
{
  _FpType __r          = _FpType(1);
  const _FpType __base = (__e < 0) ? _FpType(0.5) : _FpType(2);
  const int __n        = (__e < 0) ? -__e : __e;
  for (int __i = 0; __i < __n; ++__i)
  {
    __r *= __base;
  }
  return __r;
}
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _FpType, ::cuda::experimental::fpmp2_accuracy _Met>
class numeric_limits<::cuda::experimental::fpmp2<_FpType, _Met>>
{
private:
  // numeric_limits of the underlying IEEE-754 component type (float or double).
  using __base = numeric_limits<_FpType>;

public:
  using type = ::cuda::experimental::fpmp2<_FpType, _Met>;

  static constexpr bool is_specialized = true;

  static constexpr bool is_signed = true;
  // A non-overlapping double-word carries 2*p - 2 contiguous mantissa bits.
  static constexpr int digits = 2 * __base::digits - 2;

  // Both decimal-digit counts convert `digits` via log10(2), approximated by 30103/100000 (as CCCL's
  // float/double specializations do); the `l` suffix promotes to long so the product can't overflow int.
  //
  // digits10 = floor((digits-1)*log10(2)): most decimal digits that round-trip decimal->type->decimal.
  //     fp32mp2: floor(45*0.30103) = 13      fp64mp2: floor(103*0.30103) = 31
  static constexpr int digits10 = (digits - 1) * 30103l / 100000l;

  // max_digits10 = ceil(digits*log10(2))+1: fewest decimal digits to print any value back bit-for-bit.
  // Since digits*log10(2) is never integral, ceil(x)+1 == floor(x)+2, hence the "2 +" closed form.
  //     fp32mp2: 2 + floor(46*0.30103) = 15  fp64mp2: 2 + floor(104*0.30103) = 33
  static constexpr int max_digits10 = 2 + (digits * 30103l) / 100000l;

  // Smallest all-normal value: hi = FpType_min scaled up by p so that lo is still normal.
  _CCCL_API static constexpr type min() noexcept
  {
    return type(__base::min() * ::cuda::experimental::__fpmp_limits_exp2<_FpType>(__base::digits), _FpType(0));
  }
  // Largest value: hi = FpType_max, plus the largest lo that keeps (hi, lo) non-overlapping.
  _CCCL_API static constexpr type max() noexcept
  {
    return type(__base::max(),
                __base::max() * ::cuda::experimental::__fpmp_limits_exp2<_FpType>(-(__base::digits + 1)));
  }
  _CCCL_API static constexpr type lowest() noexcept
  {
    return type(-__base::max(),
                -(__base::max() * ::cuda::experimental::__fpmp_limits_exp2<_FpType>(-(__base::digits + 1))));
  }

  static constexpr bool is_integer = false;
  static constexpr bool is_exact   = false;
  static constexpr int radix       = __base::radix;
  _CCCL_API static constexpr type epsilon() noexcept
  {
    return type(::cuda::experimental::__fpmp_limits_exp2<_FpType>(1 - digits), _FpType(0));
  }
  _CCCL_API static constexpr type round_error() noexcept
  {
    return type(_FpType(0.5), _FpType(0));
  }

  // Exponent range of the double-word (hi + lo). The two ends are asymmetric:
  //   * Upper: magnitude is dominated by `hi`, which saturates at FpType's max, so max_exponent and
  //     max_exponent10 are inherited unchanged (fp32mp2: 128/38; fp64mp2: 1024/308).
  //   * Lower: a normalized pair needs BOTH halves normal, and `lo` sits ~p = digits(FpType) bits below
  //     `hi`; so the smallest all-normal value is 2^(emin+p), raising the floor by p binary places:
  //     min_exponent = min_exponent(FpType) + digits(FpType). (Smaller values exist but with a subnormal
  //     `lo` -- the double-word analogue of subnormals, excluded here just like float/double's own.)
  //         fp32mp2: -125 + 24 = -101  (min() = 2^-102)    fp64mp2: -1021 + 53 = -968 (min() = 2^-969)
  static constexpr int min_exponent = __base::min_exponent + __base::digits;

  // min_exponent10 = ceil((min_exponent-1)*log10(2)): most negative e with 10^e >= smallest normal.
  // Truncation toward zero coincides with ceil() for these negative products.
  //     fp32mp2: trunc(-102*0.30103) = -30      fp64mp2: trunc(-969*0.30103) = -291
  static constexpr int min_exponent10 = (min_exponent - 1) * 30103l / 100000l;

  static constexpr int max_exponent   = __base::max_exponent;
  static constexpr int max_exponent10 = __base::max_exponent10;

  static constexpr bool has_infinity                                       = true;
  static constexpr bool has_quiet_NaN                                      = true;
  static constexpr bool has_signaling_NaN                                  = true;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr float_denorm_style has_denorm = denorm_present;
  _CCCL_DEPRECATED_IN_CXX23 static constexpr bool has_denorm_loss          = false;
  _CCCL_API static constexpr type infinity() noexcept
  {
    return type(__base::infinity(), _FpType(0));
  }
  _CCCL_API static constexpr type quiet_NaN() noexcept
  {
    return type(__base::quiet_NaN(), _FpType(0));
  }
  _CCCL_API static constexpr type signaling_NaN() noexcept
  {
    return type(__base::signaling_NaN(), _FpType(0));
  }
  _CCCL_API static constexpr type denorm_min() noexcept
  {
    return type(__base::denorm_min(), _FpType(0));
  }

  // A double-word is not an IEEE-754 interchange format, but it is finite/bounded.
  static constexpr bool is_iec559  = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo  = false;

  static constexpr bool traps                    = false;
  static constexpr bool tinyness_before          = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_LIMITS_H
