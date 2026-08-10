//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_H
#define _CUDA___FP_FPMP_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp_math.h - Math Extensions for fpmp2 Types
    ======================================================================================================
    This header provides transcendental mathematical functions for fpmp2 types
    (fp32mp2 = double-float, fp64mp2 = double-double) beyond core arithmetic.
    Include this header after fpmp.h to enable mathematical functions.

    All dedicated fp32mp2 implementations use pure float-float arithmetic
    (no double-precision operations), making them suitable for GPU architectures
    where fp64 throughput is limited.

    Functions Provided:
    -------------------------------------------------------------------------
    Exponential/Logarithmic:
    - exp(x)    : Exponential function (e^x) - dedicated fp32mp2
    - log(x)    : Natural logarithm ln(x) - dedicated fp32mp2
    - log2(x)   : Base-2 logarithm - dedicated fp32mp2
    - log10(x)  : Base-10 logarithm - dedicated fp32mp2
    - log1p(x)  : Natural logarithm of (1+x) - dedicated fp32mp2
    - exp2(x)   : Base-2 exponential (2^x) - dedicated fp32mp2
    - exp10(x)  : Base-10 exponential (10^x) - dedicated fp32mp2
    - expm1(x)  : e^x - 1 - dedicated fp32mp2
    - logb(x)   : Extract exponent - placeholder

    Power:
    - pow(x,y)  : Power function x^y - dedicated fp32mp2
    - cbrt(x)   : Cube root - dedicated fp32mp2
    - rcbrt(x)  : Reciprocal cube root 1/cbrt(x) - dedicated fp32mp2

    Trigonometric:
    - sin(x)    : Sine - dedicated fp32mp2
    - cos(x)    : Cosine - dedicated fp32mp2
    - tan(x)    : Tangent - dedicated fp32mp2
    - sincos(x) : Simultaneous sine and cosine - dedicated fp32mp2
    - sinpi(x)  : sin(pix) - placeholder (host: sin(x*pi))
    - cospi(x)  : cos(pix) - placeholder (host: cos(x*pi))
    - sincospi(x): Simultaneous sin(pix) and cos(pix) - placeholder
    - asin(x)   : Arcsine - dedicated fp32mp2
    - acos(x)   : Arccosine - dedicated fp32mp2
    - atan(x)   : Arctangent - dedicated fp32mp2
    - atan2(y,x): Two-argument arctangent - dedicated fp32mp2

    Hyperbolic:
    - sinh(x)   : Hyperbolic sine - dedicated fp32mp2
    - cosh(x)   : Hyperbolic cosine - dedicated fp32mp2
    - tanh(x)   : Hyperbolic tangent - dedicated fp32mp2
    - acosh(x)  : Inverse hyperbolic cosine - dedicated fp32mp2
    - asinh(x)  : Inverse hyperbolic sine - dedicated fp32mp2
    - atanh(x)  : Inverse hyperbolic tangent - dedicated fp32mp2

    Error Functions:
    - erf(x)    : Error function - dedicated fp32mp2
    - erfc(x)   : Complementary error function - dedicated fp32mp2
    - erfcinv(x): Inverse complementary error function - placeholder (device only)
    - erfinv(x) : Inverse error function - placeholder (device only)
    - erfcx(x)  : Scaled complementary error function - placeholder (device only)

    Special Functions:
    - boys_f0(x): Boys function of zeroth order F_0(x) - dedicated fp32mp2
                  F_0(x) = 0.5*sqrt(pi/x)*erf(sqrtx),  F_0(0) = 1

    Probability / Statistics:
    - normcdfinv(p) : Inverse normal CDF Phi^-1(p) - dedicated fp32mp2
    - normcdf(x)    : Normal CDF Phi(x) - placeholder (host: 0.5*erfc(-x/sqrt2))

    Non-standard helper functions to convert integer uniform random numbers
    to Gaussian distributed random numbers:
    - icdf(uint32)  : Integer uniform -> Gaussian via normcdfinv (32-bit input)
    - icdf(uint64)  : Integer uniform -> Gaussian via normcdfinv (64-bit input)

    Gamma Functions:
    - lgamma(x) : Log-gamma function - placeholder
    - tgamma(x) : True gamma function - placeholder

    Bessel Functions (POSIX / CUDA):
    - j0(x)     : Bessel function of first kind, order 0 - placeholder
    - j1(x)     : Bessel function of first kind, order 1 - placeholder
    - jn(n,x)   : Bessel function of first kind, order n - placeholder
    - y0(x)     : Bessel function of second kind, order 0 - placeholder
    - y1(x)     : Bessel function of second kind, order 1 - placeholder
    - yn(n,x)   : Bessel function of second kind, order n - placeholder
    - cyl_bessel_i0(x) : Modified Bessel function of first kind, order 0 - placeholder
    - cyl_bessel_i1(x) : Modified Bessel function of first kind, order 1 - placeholder

    Rounding:
    - ceil(x)   : Round up to nearest integer - dedicated fp32mp2 optimization
    - floor(x)  : Round down to nearest integer - dedicated fp32mp2 optimization
    - trunc(x)  : Round toward zero - dedicated fp32mp2 optimization
    - round(x)  : Round to nearest integer - dedicated fp32mp2 optimization
    - rint(x)   : Round using current rounding mode - placeholder
    - nearbyint(x): Round using current rounding mode (no FE exception) - placeholder
    - lrint(x)  : Round to long int - placeholder
    - lround(x) : Round to long int (away from zero) - placeholder
    - llrint(x) : Round to long long int - placeholder
    - llround(x): Round to long long int (away from zero) - placeholder

    Floating-Point Manipulation:
    - fabs(x)   : Absolute value - dedicated fp32mp2 optimization (operates on (hi, lo) directly)
    - copysign(x,y): Copy sign of y to x - placeholder
    - ldexp(x,n): x * 2^n - dedicated fp32mp2
    - scalbn(x,n): x * FLT_RADIX^n - dedicated fp32mp2 (forwards to ldexp)
    - scalbln(x,n): x * FLT_RADIX^n (long exponent) - placeholder
    - frexp(x,*n): Extract mantissa and exponent - placeholder
    - modf(x,*i): Split into integer and fractional parts - placeholder
    - logb(x)   : Extract exponent - placeholder
    - ilogb(x)  : Extract exponent as int - placeholder
    - nextafter(x,y): Next representable value - placeholder

    Min/Max/Difference:
    - fmax(x,y) : Maximum - dedicated fp32mp2 optimization (lexicographic compare on (hi, lo), NaN-aware)
    - fmin(x,y) : Minimum - dedicated fp32mp2 optimization (lexicographic compare on (hi, lo), NaN-aware)
    - max(x,y)  : Maximum - dedicated fp32mp2 optimization (std::max-like, first-arg tie/unordered behavior)
    - min(x,y)  : Minimum - dedicated fp32mp2 optimization (std::min-like, first-arg tie/unordered behavior)
    - fdim(x,y) : Positive difference - placeholder

    Remainder:
    - fmod(x,y) : Floating-point remainder - dedicated fp32mp2
    - remainder(x,y): IEEE remainder - dedicated fp32mp2
    - remquo(x,y,*q): Remainder with quotient bits - placeholder

    Distance:
    - hypot(x,y): Hypotenuse sqrt(x^2+y^2) - placeholder
    - rhypot(x,y): Reciprocal hypotenuse - placeholder (host: 1/hypot)

    Vector Norms:
    - norm3d(a,b,c)    : 3D Euclidean norm sqrt(a^2+b^2+c^2) - placeholder
    - norm4d(a,b,c,d)  : 4D Euclidean norm sqrt(a^2+b^2+c^2+d^2) - placeholder
    - rnorm3d(a,b,c)   : Reciprocal 3D norm - placeholder
    - rnorm4d(a,b,c,d) : Reciprocal 4D norm - placeholder

    Classification (portable prefixed API + conditional standard overloads):
    - fpmp_isfinite(x): Test for finite value - placeholder
    - fpmp_isinf(x)   : Test for infinity - placeholder
    - fpmp_isnan(x)   : Test for NaN - placeholder
    - fpmp_signbit(x) : Test sign bit - placeholder
      Standard names isfinite/isinf/isnan/signbit are exposed only when the
      corresponding macro is not defined.

    Warp Shuffle (CUDA-only, modern __shfl_sync family): the fpmp2 overloads
    of __shfl_sync / __shfl_xor_sync / __shfl_down_sync / __shfl_up_sync are
    thread-cooperation primitives, not math, so they live in the core header
    <cuda/__fp/fpmp.h> (available via <cuda/fpmp>), not here.

    Dedicated Implementation Details:
    -------------------------------------------------------------------------
    exp(x) for fp32mp2:
      Argument reduction: x = n*ln2 + r, |r| < ln2/2.
      Core: 14-term Taylor series for exp(r) in fp32mp2.
      Reconstruction: result * 2^n via IEEE-754 exponent manipulation.
      Accuracy: ~10^-10 - 10^-11 relative error.

    log(x) for fp32mp2:
      Range reduction: x = m*2^e, m  in  [1, sqrt2).
      Core: log(m) = 2*atanh((m-1)/(m+1)) via degree-8 minimax polynomial.
      Reconstruction: log(x) = log(m) + e*ln2 with fp32mp2 ln2 constant.
      Handles denormals via pre-scaling.

    log1p(x) for fp32mp2:
      Forwards to log(1+x) after computing 1+x via fp32mp2 2-sum, which
      preserves the small-x lo (the lo carries x exactly when |x| <= 1
      causes float hi cancellation).  log()'s accurate sub for (m-1)
      then recovers full fp32mp2 precision in the asinh-form Horner
      core, yielding the a - a^2/2 + a^3/3 - ... expansion implicitly.
      Special cases (x = -1 -> -inf, x < -1 -> NaN, +-inf) handled
      explicitly before forwarding.

    log2(x), log10(x) for fp32mp2:
      Composition over the dedicated fp32mp2 natural log:
        log2(x)  = log(x) * (1/ln 2)
        log10(x) = log(x) * (1/ln 10)
      with the reciprocal stored as a single fp32mp2 (hi+lo) constant.
      The one ff-multiply costs ~1 ulp on the lo limb; combined with
      the ~46-bit precision of __fpmp2_log this still leaves >44
      bits of accuracy across the whole representable input range,
      matching the fp32mp2 noise floor.  All domain special cases
      (x<=0, +-inf, NaN) are handled inside __fpmp2_log.

    exp2(x), exp10(x) for fp32mp2:
      Dedicated implementations following a "single base-2
      split" strategy.  Pseudocode:
        t      = x * log2(base)        [exp2: t = x; exp10: t = x * log2 10]
        n      = round(t.hi)           [exact integer = binary exponent]
        r      = t - n                 [|r.hi| <= 0.5]
        2^r    via the inlined base-2 Taylor kernel
               `__internal_fpmp2_exp2_kernel` with coefficients
               a_k = (ln 2)^k / k!     [no r * ln 2 detour, no
                                        natural-log reduction inside]
        result = 2^n * 2^r             [via split-exponent helper]
      The single integer split happens in *base-2* units, so the 2^n
      factor drops out exactly and the kernel never touches a value
      outside [-0.5, 0.5].  Compared to the earlier composition
      exp(x * ln base) -- which forced __fpmp2_exp to re-derive
      n_internal from the already-amplified product, stacking two
      reduction errors -- the dedicated path keeps only one __ffloat
      multiplication on the input side (and zero for exp2).  Net:
      `exp2` matches the composed path within 1 bit on the `work`
      dataset while cutting fp32mp2 call cost; `exp10` matches the
      composed path's 39-bit `work` accuracy at lower instruction
      count.  Overflow / underflow shortcuts (|x| at the float
      exponent boundary) avoid the wasted polynomial / scaling work
      when the result is already a rounded +-inf / 0.

    expm1(x) for fp32mp2:
      Strategy mirrors log1p:
        |x_hi| < 1/2:  direct Taylor series
                       expm1(x) = x + x^2 * (1/2 + x/6 + x^2/24 + ...)
                       evaluated as 12 mixed-precision Horner terms
                       (M = 4 split, top 4 plain float, bottom 8 ff).
        |x_hi| >= 1/2: exp(x) - 1 via fp32mp2 accurate sub against
                       the constant 1.0; the subtraction loses <= 1
                       bit because |exp(x) - 1| >= 0.4 in this band.
      The polynomial branch covers ~25 % of a normal-around-0 input
      distribution; warp divergence cost stays modest.  Truncation
      noise at the branch point: omitted x^13 term ~= 0.5^13/13!
      ~= 1.96*10^-14, well below fp32mp2 ulp at expm1(1/2).
      Special cases: NaN->NaN, +inf->+inf, -inf->-1, x=0->0.

    asinh(x), acosh(x), atanh(x) for fp32mp2:
      Inverse hyperbolic family -- all three reduce to a single log1p
      call with cancellation-safe arithmetic forms:
        asinh(x) = sign(x)*log1p(|x| + x^2/(sqrt(x^2+1)+1))
        acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))     for x >= 1
        atanh(x) = 0.5*sign(x)*log1p(2|x|/(1-|x|))      for |x| >= 0.25
        atanh(x) = x*(1 + y*P(y)),  y = x^2,             for |x| < 0.25
                   P(y) = 1/3 + y/5 + y^2/7 + ... + y^11/25
      The rationalized arguments avoid the (1 - cos)-style cancellation
      that breaks the textbook formulas around x = 0 (asinh, atanh)
      and x = 1 (acosh).  acosh's `(x-1)*(x+1)` factorization sidesteps
      the catastrophic cancellation that `x^2-1` would suffer near 1.
      atanh's polynomial branch covers ~25 % of the typical work range:
      it bypasses the divide-driven precision loss in the log1p form
      around 0 while keeping warp divergence small.  Large-|x| paths
      (asinh, acosh) switch to log(2|x|) above 2^25 -- early enough
      that the lo-limb ulp accumulation in the (x^2+1)/(x-1)(x+1) chain
      doesn't degrade precision, and the dropped 1/(4x^2) correction
      sits below fp32mp2 ulp throughout the asymptotic region.

    ldexp(x, n) for fp32mp2:
      result = x * 2^n built directly in fp32 via bit-cast of the
      biased exponent.  No fp64 round-trip -- important on GPUs where
      double-precision throughput is 1:32 of float.  The 2^n factor is
      split into three pieces  2^k * 2^k * 2^(n - 2k)  (k = n/3) so each
      factor's biased exponent stays inside the fp32 normal range
      [1, 254]; this avoids the spurious 0*inf = NaN that a single
      saturated scale would produce when x = 0 and n is large.  n is
      pre-clamped to +-300, which is wider than any input range that can
      actually round to something other than +-0 or +-inf: 2^300 overflows
      every fp32 (denormals included), and 2^-300 underflows every
      finite fp32.  Special cases (NaN/+-inf/+-0) propagate naturally through
      the three multiplications with the lo limb cleared.

    fmod(x, y), remainder(x, y) for fp32mp2:
      Integer-mantissa long division __nv_fmod.
      Each operand is decomposed into a 64-bit mantissa M
      and a binary exponent E (value = M * 2^E, M normalized to
      [2^52, 2^53)), then  Mx * 2^(Ex-Ey) mod My  is evaluated with
      exact uint64 arithmetic, chunking the left shift by 10 bits so the
      running dividend never exceeds 2^63 (~D/10 iterations).  No
      fp64 is touched, so the routine keeps full fp32 throughput.  The
      53-bit window matches IEEE double's significand: a renormalized
      fp32mp2 carries 24 (hi) + 24 (lo) significant bits whose full span
      is ~49 bits, so the uint64 mantissa is exact and the result equals
      the former ::fmod(double,double) round-trip bit-for-bit - but
      without any fp64 instructions.  (A 48-bit window rounded away the
      bottom of lo, which fmod's inherent cancellation then amplified to
      ~27-bit results.)  remainder adds the round-to-
      nearest-even step (compare 2*ia vs My, ties broken by the parity of
      the accumulated quotient) and the |x| < |y| short-circuit.  Special
      cases: NaN or x = +-inf or y = 0 -> NaN; y = +-inf -> x.

    erf(x) for fp32mp2:
      erf(x) = -expm1(-|x|*P(|x|)).
      Core: degree-24 Remez polynomial P for the argument, followed by
      expm1 via argument reduction + polynomial, all in fp32mp2.

    erfc(x) for fp32mp2:
      erfc(x) = erfcx(|x|)*exp(-x^2).
      Core: erfcx approximated by a degree-22 Chebyshev polynomial in
      the transformed variable t = 1/(1+|x|), combined with a dedicated
      exp(-x^2) evaluation using a degree-8 polynomial, all in fp32mp2.
      Uses the identity erfc(-x) = 2 - erfc(x) for negative arguments.

    normcdfinv(p) for fp32mp2:
      Rational approximation (Mike Giles coefficients).
      Variable: w = -log(4p(1-p)), computed via fp32mp2 log.
      Central branch (w < 6.125): degree-22 Horner polynomial in (w - 3.125).
      Tail 1 branch (6.125 <= w < 16): degree-18 Horner polynomial in (sqrtw - 3.25).
      Tail 2 branch (w >= 16): degree-24 Horner polynomial in (sqrtw - 7.25),
        covering all representable float inputs including denormals,
        with full fp32mp2 precision (~46+ bits) across the entire range.
      Returns +-infinity for p <= 0 or p >= 1 (standard mathematical convention).
      The icdf() wrappers clamp to +-FLT_MAX for safe Gaussian variate generation.

    icdf(uint32_t x) for fp32mp2:
      Converts a 32-bit integer uniform RNG output to a Gaussian sample.
      Mirrors x around 2^31 to map into (0, 0.5] (cuRAND convention).
      Computes p = (x + 0.5)/2^32 as an exact fp32mp2 value by splitting
      x into two 16-bit halves, then calls normcdfinv(p).

    icdf(uint64_t x) for fp32mp2:
      64-bit variant.  Keeps the top 48 bits (matching fp32mp2 precision),
      mirrors around 2^47, computes p = (x + 0.5)/2^48 by splitting into
      two 24-bit halves, then calls normcdfinv(p).

    sin(x), cos(x), sincos(x) for fp32mp2:
      Argument reduction: x = n*(pi/2) + r, |r| <= pi/4.
      Three paths based on |x|:
        Tiny (|x| < pi/4):  no reduction needed.
        Fast (|x| < 2^20):  Cody-Waite with 3-piece pi/2 (~70 bits),
          error-tracked via two_mult_fma + two_sum.
        Large (|x| >= 2^20): Payne-Hanek using integer 2/pi table (160 bits),
          returning fp32mp2 via extended-precision fixed-point conversion
          without any fp64 operations.
      Core: sin(r) via 8-term Taylor (x through x^15),
            cos(r) via 9-term Taylor (1 through x^16),
            both evaluated in fp32mp2 Horner form.
      sincos computes both kernels; sin/cos call sincos internally.
      Quadrant mapping via n mod 4 with sign/swap adjustment.

    Placeholder functions:
    - Delegate to standard double-precision system functions
    - Intended as API stubs; they do not provide full multi-precision accuracy yet

    fp64mp2 implementations:
    - When _CCCL_FPMP_FP128_MATH_FALLBACK=1, route through __float128 (libquadmath
      on host, CUDA fp128 intrinsics on device) for ~113-bit accuracy.
    - When _CCCL_FPMP_FP128_MATH_FALLBACK=0, fall back to double-precision math.
    - The fp64mp2 normcdfinv uses CUDA erfcinv on device, the fp32mp2 polynomial
      on host (no standard erfcinv available).

    Configuration Macros:
    -------------------------------------------------------------------------
    Which of the two fp64mp2 paths above a build gets is decided by
    _CCCL_FPMP_FP128_MATH_FALLBACK, defined and documented in fpmp_math_impl.h together with
    _CCCL_FPMP_FP128_QUAD_ERF, which says whether the erf family joins them. Both follow
    _CCCL_FPMP_FP128_ENABLE and _CCCL_FPMP_FP128_DEVICE_OPS from fpmp_impl.h.

    Left alone, that switch is decided per compilation pass: a host-only build takes the
    quad path wherever fp128 is available, but in a CUDA compilation only the device pass
    does, and only where every targeted architecture can run fp128 (sm_100 and later). A
    .cu file therefore does not silently acquire a libquadmath dependency its host-only
    counterpart never had, at the price of the two halves differing in accuracy.

    Programs whose host and device results have to agree to the last bits put both passes
    on the quad path with the public knob CCCL_FPMP_FP128_MATH_FALLBACK:

      nvcc -arch=sm_100 -DCCCL_FPMP_FP128_MATH_FALLBACK=1 app.cu -lquadmath

    -lquadmath is the host side of that bargain on x86_64 GCC, where the quad entry points
    (expq, sinq, ...) live in libquadmath; hosts whose long double is IEEE binary128
    (AArch64, PPC64LE, s390x) call libm's *l entry points and need nothing extra. Asking
    for it on a target whose device cannot run fp128 makes the device pass fail to compile,
    since its bodies then need quad arithmetic the architecture does not have. Every
    translation unit has to agree on the value, as does the library build in library mode.
*/
#include <cuda/__fp/fpmp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

// Header-mode implementations are provided by the per-family implementation
// headers below (see docs/libcudacxx/fp/fpmp_spec.rst, "Function Families").
// In library mode (_CCCL_FPMP_USE_LIB) the kernels come from the compiled
// library and only the declarations in the #else branch below are used.
#if !(defined _CCCL_FPMP_USE_LIB)
#  include <cuda/__fp/fpmp_math_impl.h>
#  include <cuda/__fp/fpmp_math_impl_classify.h>
#  include <cuda/__fp/fpmp_math_impl_exp.h>
#  include <cuda/__fp/fpmp_math_impl_hyperbolic.h>
#  include <cuda/__fp/fpmp_math_impl_manip.h>
#  include <cuda/__fp/fpmp_math_impl_nearint.h>
#  include <cuda/__fp/fpmp_math_impl_pow.h>
#  include <cuda/__fp/fpmp_math_impl_special.h>
#  include <cuda/__fp/fpmp_math_impl_trig.h>
/* Cleanup: undefine the fallback-body and dispatch factory macros so they don't
 * leak into headers/translation units that include this file. */
#  undef _CCCL_FPMP_MATH_FALLBACK_1A
#  undef _CCCL_FPMP_MATH_FALLBACK_2A
#  undef _CCCL_FPMP_MATH_FALLBACK_1A_RETLL
#  undef _CCCL_FPMP_MATH_FALLBACK_1A_RETL
#  undef _CCCL_FPMP_MATH_ONLY_FP32_FP64
#  undef _CCCL_FPMP_MATH_DISPATCH_1A
#  undef _CCCL_FPMP_MATH_DISPATCH_2A
#  undef _CCCL_FPMP_MATH_DISPATCH_2A_YX
#  undef _CCCL_FPMP_MATH_DISPATCH_3A
#  undef _CCCL_FPMP_MATH_DISPATCH_4A
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_RETINT
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_RETLL
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_RETL
#  undef _CCCL_FPMP_MATH_DISPATCH_1A_2OUT
#  undef _CCCL_FPMP_MATH_DISPATCH_INT_FP
#  undef _CCCL_FPMP_MATH_DISPATCH_2A_QUO
#endif // !_CCCL_FPMP_USE_LIB

#if (defined _CCCL_FPMP_USE_LIB)
#  include <cuda/__fp/fpmp_math_impl_lib.h>
#endif // _CCCL_FPMP_USE_LIB

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/*
 * ============================================================================
 * Freestanding API functions for fpmp2 class
 * ============================================================================
 */

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> exp(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_exp(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> log(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_log(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> log2(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_log2(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> log10(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_log10(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> log1p(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_log1p(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
pow(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_pow(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> cbrt(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_cbrt(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> sin(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_sin(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> cos(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_cos(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline void
sincos(const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __s, fpmp2<_FpType, _TypeAcc>* __c) noexcept
{
  _FpType __sin_hi, __sin_lo, __cos_hi, __cos_lo;
  __fpmp2_sincos(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo);
  *__s = fpmp2<_FpType, _TypeAcc>(__sin_hi, __sin_lo);
  *__c = fpmp2<_FpType, _TypeAcc>(__cos_hi, __cos_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> asin(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_asin(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> acos(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_acos(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> atan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_atan(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
atan2(const fpmp2<_FpType, _TypeAcc>& __y, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_atan2(__y.hi(), __y.lo(), __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> sinh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_sinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> cosh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_cosh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> tanh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_tanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> erf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_erf(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> erfc(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_erfc(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> boys_f0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_boys_f0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> normcdfinv(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_normcdfinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// icdf takes no fpmp2 argument, so the accuracy tag of the result cannot be deduced
// and is the one place in this header that still defaults, like the class template.
template <fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_HOST_DEVICE_API inline fpmp2<float, _TypeAcc> icdf(uint32_t __x) noexcept
{
  float __res_hi;
  float __res_lo;
  __fpmp2_icdf(__x, &__res_hi, &__res_lo);
  return fpmp2<float, _TypeAcc>(__res_hi, __res_lo);
}

template <fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_HOST_DEVICE_API inline fpmp2<float, _TypeAcc> icdf(uint64_t __x) noexcept
{
  float __res_hi;
  float __res_lo;
  __fpmp2_icdf(__x, &__res_hi, &__res_lo);
  return fpmp2<float, _TypeAcc>(__res_hi, __res_lo);
}

// Inverse hyperbolic functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> acosh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_acosh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> asinh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_asinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> atanh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_atanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Tangent
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> tan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_tan(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Additional exponential/logarithmic functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> exp2(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_exp2(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> exp10(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_exp10(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> expm1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_expm1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> logb(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_logb(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Rounding functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> ceil(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_ceil(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> floor(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_floor(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> trunc(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_trunc(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> round(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_round(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> rint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_rint(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> nearbyint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_nearbyint(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Absolute value
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> fabs(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_fabs(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Gamma functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> lgamma(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_lgamma(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> tgamma(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_tgamma(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Bessel functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> j0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_j0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> j1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_j1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> y0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_y0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> y1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_y1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> jn(int __n, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_jn(__n, __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> yn(int __n, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_yn(__n, __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> cyl_bessel_i0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_cyl_bessel_i0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> cyl_bessel_i1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_cyl_bessel_i1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// CUDA-specific trigonometric functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> sinpi(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_sinpi(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> cospi(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_cospi(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline void
sincospi(const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __s, fpmp2<_FpType, _TypeAcc>* __c) noexcept
{
  _FpType __sin_hi, __sin_lo, __cos_hi, __cos_lo;
  __fpmp2_sincospi(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo);
  *__s = fpmp2<_FpType, _TypeAcc>(__sin_hi, __sin_lo);
  *__c = fpmp2<_FpType, _TypeAcc>(__cos_hi, __cos_lo);
}

// Normal distribution CDF and reciprocal functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> normcdf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_normcdf(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> rcbrt(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_rcbrt(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> erfcinv(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_erfcinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> erfinv(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_erfinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> erfcx(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_erfcx(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
norm3d(const fpmp2<_FpType, _TypeAcc>& __a,
       const fpmp2<_FpType, _TypeAcc>& __b,
       const fpmp2<_FpType, _TypeAcc>& __c) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_norm3d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
norm4d(const fpmp2<_FpType, _TypeAcc>& __a,
       const fpmp2<_FpType, _TypeAcc>& __b,
       const fpmp2<_FpType, _TypeAcc>& __c,
       const fpmp2<_FpType, _TypeAcc>& __d) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_norm4d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), __d.hi(), __d.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
rnorm3d(const fpmp2<_FpType, _TypeAcc>& __a,
        const fpmp2<_FpType, _TypeAcc>& __b,
        const fpmp2<_FpType, _TypeAcc>& __c) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_rnorm3d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
rnorm4d(const fpmp2<_FpType, _TypeAcc>& __a,
        const fpmp2<_FpType, _TypeAcc>& __b,
        const fpmp2<_FpType, _TypeAcc>& __c,
        const fpmp2<_FpType, _TypeAcc>& __d) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_rnorm4d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), __d.hi(), __d.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Two-argument functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
fmax(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_fmax(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
fmin(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_fmin(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
max(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_max(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
min(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_min(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
fmod(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_fmod(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
remainder(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_remainder(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
hypot(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_hypot(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
copysign(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_copysign(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
fdim(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_fdim(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
nextafter(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_nextafter(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
rhypot(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_rhypot(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Functions with special signatures
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
remquo(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y, int* __quo) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_remquo(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo, __quo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> ldexp(const fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_ldexp(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> scalbn(const fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_scalbn(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> scalbln(const fpmp2<_FpType, _TypeAcc>& __x, long int __n) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_scalbln(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc> frexp(const fpmp2<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{
  _FpType __res_hi, __res_lo;
  __fpmp2_frexp(__x.hi(), __x.lo(), &__res_hi, &__res_lo, __nptr);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline fpmp2<_FpType, _TypeAcc>
modf(const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __iptr) noexcept
{
  _FpType __res_hi, __res_lo, __i_hi, __i_lo;
  __fpmp2_modf(__x.hi(), __x.lo(), &__res_hi, &__res_lo, &__i_hi, &__i_lo);
  *__iptr = fpmp2<_FpType, _TypeAcc>(__i_hi, __i_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Functions returning integer types
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int ilogb(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_ilogb(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline long long int llrint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_llrint(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline long long int llround(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_llround(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline long int lrint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_lrint(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline long int lround(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_lround(__x.hi(), __x.lo());
}

// Classification functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int fpmp_isfinite(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_isfinite(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int fpmp_isinf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_isinf(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int fpmp_isnan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_isnan(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int fpmp_signbit(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return __fpmp2_signbit(__x.hi(), __x.lo());
}

// The same tests under their standard names.
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int isfinite(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_isfinite(__x);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int isinf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_isinf(__x);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int isnan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_isnan(__x);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline int signbit(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_signbit(__x);
}

/*
 * Note: the fpmp2 warp-shuffle overloads (__shfl_sync, __shfl_xor_sync,
 * __shfl_down_sync, __shfl_up_sync) are thread-cooperation primitives, not math
 * functions, so they live in the core header <cuda/__fp/fpmp.h> (available via
 * <cuda/fpmp>) rather than here.
 */
} // namespace cuda::experimental

// ============================================================================
// cuda::std overloads for the standard <cmath> names.
//
// The emulated math lives in cuda::experimental, but a qualified
// cuda::std::<fn>(x) call suppresses ADL, so without these overloads it would
// silently narrow fpmp2 -> double (via the implicit conversion) and compute a
// native-double result. These forward to the cuda::experimental implementations
// (which unqualified / ADL calls already resolve to). Only names that cuda::std
// actually declares are provided; the CUDA-only extensions (rsqrt, exp10,
// rcbrt, sinpi, cospi, j0/j1/y0/y1, cyl_bessel_*, normcdf*, erf*inv, erfcx,
// norm*/rnorm*/rhypot, ...) have no cuda::std counterpart and are omitted.
// ============================================================================
_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _CCCL_FPMP_STD_UNARY(_Name)                                           \
  template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>     \
  _CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc> _Name( \
    const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept       \
  {                                                                           \
    return ::cuda::experimental::_Name(__x);                                  \
  }

_CCCL_FPMP_STD_UNARY(exp)
_CCCL_FPMP_STD_UNARY(exp2)
_CCCL_FPMP_STD_UNARY(expm1)
_CCCL_FPMP_STD_UNARY(log)
_CCCL_FPMP_STD_UNARY(log2)
_CCCL_FPMP_STD_UNARY(log10)
_CCCL_FPMP_STD_UNARY(log1p)
_CCCL_FPMP_STD_UNARY(logb)
_CCCL_FPMP_STD_UNARY(cbrt)
_CCCL_FPMP_STD_UNARY(sin)
_CCCL_FPMP_STD_UNARY(cos)
_CCCL_FPMP_STD_UNARY(tan)
_CCCL_FPMP_STD_UNARY(asin)
_CCCL_FPMP_STD_UNARY(acos)
_CCCL_FPMP_STD_UNARY(atan)
_CCCL_FPMP_STD_UNARY(sinh)
_CCCL_FPMP_STD_UNARY(cosh)
_CCCL_FPMP_STD_UNARY(tanh)
_CCCL_FPMP_STD_UNARY(asinh)
_CCCL_FPMP_STD_UNARY(acosh)
_CCCL_FPMP_STD_UNARY(atanh)
_CCCL_FPMP_STD_UNARY(erf)
_CCCL_FPMP_STD_UNARY(erfc)
_CCCL_FPMP_STD_UNARY(tgamma)
_CCCL_FPMP_STD_UNARY(lgamma)
_CCCL_FPMP_STD_UNARY(ceil)
_CCCL_FPMP_STD_UNARY(floor)
_CCCL_FPMP_STD_UNARY(trunc)
_CCCL_FPMP_STD_UNARY(round)
_CCCL_FPMP_STD_UNARY(rint)
_CCCL_FPMP_STD_UNARY(nearbyint)
_CCCL_FPMP_STD_UNARY(fabs)

#undef _CCCL_FPMP_STD_UNARY

#define _CCCL_FPMP_STD_BINARY(_Name)                                          \
  template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>     \
  _CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc> _Name( \
    const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x,                \
    const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __y) noexcept       \
  {                                                                           \
    return ::cuda::experimental::_Name(__x, __y);                             \
  }

_CCCL_FPMP_STD_BINARY(pow)
_CCCL_FPMP_STD_BINARY(atan2)
_CCCL_FPMP_STD_BINARY(fmod)
_CCCL_FPMP_STD_BINARY(remainder)
_CCCL_FPMP_STD_BINARY(hypot)
_CCCL_FPMP_STD_BINARY(fmax)
_CCCL_FPMP_STD_BINARY(fmin)
_CCCL_FPMP_STD_BINARY(copysign)
_CCCL_FPMP_STD_BINARY(fdim)
_CCCL_FPMP_STD_BINARY(nextafter)

#undef _CCCL_FPMP_STD_BINARY

#define _CCCL_FPMP_STD_UNARY_RET(_Ret, _Name)                                                          \
  template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>                              \
  _CCCL_HOST_DEVICE_API _Ret _Name(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept \
  {                                                                                                    \
    return ::cuda::experimental::_Name(__x);                                                           \
  }

_CCCL_FPMP_STD_UNARY_RET(int, ilogb)
_CCCL_FPMP_STD_UNARY_RET(long long int, llrint)
_CCCL_FPMP_STD_UNARY_RET(long long int, llround)
_CCCL_FPMP_STD_UNARY_RET(long int, lrint)
_CCCL_FPMP_STD_UNARY_RET(long int, lround)

#undef _CCCL_FPMP_STD_UNARY_RET

// Functions with special signatures (extra scalar / out-pointer arguments).
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
ldexp(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return ::cuda::experimental::ldexp(__x, __n);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
scalbn(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return ::cuda::experimental::scalbn(__x, __n);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
scalbln(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x, long int __n) noexcept
{
  return ::cuda::experimental::scalbln(__x, __n);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
frexp(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{
  return ::cuda::experimental::frexp(__x, __nptr);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
modf(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x,
     ::cuda::experimental::fpmp2<_FpType, _TypeAcc>* __iptr) noexcept
{
  return ::cuda::experimental::modf(__x, __iptr);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2<_FpType, _TypeAcc>
remquo(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x,
       const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __y,
       int* __quo) noexcept
{
  return ::cuda::experimental::remquo(__x, __y, __quo);
}

// Classification functions.
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API int isfinite(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::experimental::isfinite(__x);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API int isinf(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::experimental::isinf(__x);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API int isnan(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::experimental::isnan(__x);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API int signbit(const ::cuda::experimental::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::experimental::signbit(__x);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_H
