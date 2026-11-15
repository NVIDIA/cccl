//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_H
#define _CUDA___FP_FPMP_MATH_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
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
    It is reached through <cuda/fpmp>, which includes it after fpmp.h; it is not
    an entry point of its own.

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
    - logb(x)   : Extract exponent - exact on the limb pair

    Power:
    - pow(x,y)  : Power function x^y - dedicated fp32mp2
    - cbrt(x)   : Cube root - dedicated fp32mp2
    - rcbrt(x)  : Reciprocal cube root 1/cbrt(x) - dedicated fp32mp2

    Trigonometric:
    - sin(x)    : Sine - dedicated fp32mp2
    - cos(x)    : Cosine - dedicated fp32mp2
    - tan(x)    : Tangent - dedicated fp32mp2
    - sincos(x) : Simultaneous sine and cosine - dedicated fp32mp2
    - sinpi(x)  : sin(pix) - dedicated fp32mp2
    - cospi(x)  : cos(pix) - dedicated fp32mp2
    - sincospi(x): Simultaneous sin(pix) and cos(pix) - dedicated fp32mp2
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
    - erfcinv(x): Inverse complementary error function - dedicated fp32mp2
    - erfinv(x) : Inverse error function - dedicated fp32mp2
    - erfcx(x)  : Scaled complementary error function - dedicated fp32mp2

    Special Functions:
    - boys_f0(x): Boys function of zeroth order F_0(x) - dedicated fp32mp2
                  F_0(x) = 0.5*sqrt(pi/x)*erf(sqrtx),  F_0(0) = 1

    Probability / Statistics:
    - normcdfinv(p) : Inverse normal CDF Phi^-1(p) - dedicated fp32mp2
    - normcdf(x)    : Normal CDF Phi(x) - dedicated fp32mp2

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
    - rint(x)   : Round to nearest, ties to even - exact on the limb pair
    - nearbyint(x): Round to nearest, ties to even (no FE exception) - exact on the limb pair
    - lrint(x)  : Round to long int - dedicated fp32mp2 (exact, over rint)
    - lround(x) : Round to long int (away from zero) - dedicated fp32mp2 (exact, over round)
    - llrint(x) : Round to long long int - dedicated fp32mp2 (exact, over rint)
    - llround(x): Round to long long int (away from zero) - dedicated fp32mp2 (exact, over round)

    Floating-Point Manipulation:
    - fabs(x)   : Absolute value - dedicated fp32mp2 optimization (operates on (hi, lo) directly)
    - copysign(x,y): Copy sign of y to x - exact on the limb pair
    - ldexp(x,n): x * 2^n - dedicated fp32mp2
    - scalbn(x,n): x * FLT_RADIX^n - dedicated fp32mp2 (forwards to ldexp)
    - scalbln(x,n): x * FLT_RADIX^n (long exponent) - exact (forwards to ldexp)
    - frexp(x,*n): Extract mantissa and exponent - exact on the limb pair
    - modf(x,*i): Split into integer and fractional parts - exact on the limb pair
    - logb(x)   : Extract exponent - exact on the limb pair
    - ilogb(x)  : Extract exponent as int - exact on the limb pair
    - nextafter(x,y): Next representable value - placeholder

    Min/Max/Difference:
    - fmax(x,y) : Maximum - dedicated fp32mp2 optimization (lexicographic compare on (hi, lo), NaN-aware)
    - fmin(x,y) : Minimum - dedicated fp32mp2 optimization (lexicographic compare on (hi, lo), NaN-aware)
    - max(x,y)  : Maximum - dedicated fp32mp2 optimization (std::max-like, first-arg tie/unordered behavior)
    - min(x,y)  : Minimum - dedicated fp32mp2 optimization (std::min-like, first-arg tie/unordered behavior)
    - fdim(x,y) : Positive difference - dedicated fp32mp2 (exact compare, accurate difference)

    Remainder:
    - fmod(x,y) : Floating-point remainder - dedicated fp32mp2
    - remainder(x,y): IEEE remainder - dedicated fp32mp2
    - remquo(x,y,*q): Remainder with quotient bits - dedicated fp32mp2 (remainder's reduction)

    Distance:
    - hypot(x,y): Hypotenuse sqrt(x^2+y^2) - dedicated fp32mp2
    - rhypot(x,y): Reciprocal hypotenuse 1/hypot(x,y) - dedicated fp32mp2

    Vector Norms:
    - norm3d(a,b,c)    : 3D Euclidean norm sqrt(a^2+b^2+c^2) - placeholder
    - norm4d(a,b,c,d)  : 4D Euclidean norm sqrt(a^2+b^2+c^2+d^2) - placeholder
    - rnorm3d(a,b,c)   : Reciprocal 3D norm - placeholder
    - rnorm4d(a,b,c,d) : Reciprocal 4D norm - placeholder

    Classification (portable prefixed API + conditional standard overloads):
    - fpmp_isfinite(x): Test for finite value - exact on the limb pair (both limbs)
    - fpmp_isinf(x)   : Test for infinity - exact on the limb pair (both limbs)
    - fpmp_isnan(x)   : Test for NaN - exact on the limb pair (both limbs)
    - fpmp_signbit(x) : Test sign bit - exact on the limb pair (both limbs)
      Standard names isfinite/isinf/isnan/signbit are exposed only when the
      corresponding macro is not defined.

    Warp Shuffle (CUDA-only, modern __shfl_sync family): the fpmp2 overloads
    of __shfl_sync / __shfl_xor_sync / __shfl_down_sync / __shfl_up_sync are
    thread-cooperation primitives, not math, so they live in the core header
    <cuda/__fp/fpmp.h> (available via <cuda/fpmp>), not here.

    Dedicated Implementation Details:
    -------------------------------------------------------------------------
    exp(x), exp2(x), exp10(x), expm1(x) for fp32mp2:
      Table driven, on the tables of pow. x is reduced exactly to a
      multiple k of ln2/128 (of 1/128 for exp2, of log10(2)/128 for
      exp10) plus a pair phi, |phi| <= ln2/256, by a three-part
      Cody-Waite step whose leading constant has 9 bits, so that the
      error does not grow with |x|:
        b^x = 2^n * 2^(j/128) * e^phi,   k = 128 n + j
      2^(j/128) comes from a 128-entry table and e^phi - 1 from a
      degree-5 polynomial; the scaling by 2^n is applied in two steps,
      so that a subnormal result is rounded once.  exp2 of an integer
      is the exact power of two.
      expm1 takes j from [-64, 63], 2^(j/128) - 1 from a table of its
      own and e^phi - 1 to full relative accuracy, and returns
      (2^n - 1) + 2^n (2^(j/128) e^phi - 1); the sum is the result
      itself for |x| < 0.35, where n = 0.  fpmp2_accuracy::high: below
      2^-7 in magnitude it is the Taylor series x + x^2 P(x) in pair
      arithmetic instead, which avoids the cancellation in the buckets
      next to zero (there 2.4u^2 against 4.4u^2).
      Accuracy against MPFR: within 3.2u^2 (u = 2^-24), 4.5u^2 for expm1
      (3.9u^2 at high).
      Special cases: NaN->NaN, overflow->+inf, underflow->+0 (-1 for
      expm1), expm1(+-0) = +-0.

    log(x), log2(x), log10(x), log1p(x) for fp32mp2:
      Table driven, on the table of pow. x = 2^k z with z in
      [0.701, 1.402); the bucket holding z, one of 128, gives 1/c with
      9 bits, so that r = z/c - 1 is exact, and ln(c) as three floats:
        ln(x) = k ln2 + ln(c) + r + r^2 (-1/2 + r P(r)),   |r| < 2^-7.5
      The bucket around 1 has c = 1, which keeps the result relatively
      accurate as x -> 1.  No division is involved.
      log2 and log10 scale ln(z) and add k, or k log10(2), afterwards;
      log2 of a power of two is exact.  log1p forms r from x itself
      where 1 + x is in [0.701, 1.402), so that the result keeps the
      relative accuracy of x as x -> 0, and otherwise takes the
      logarithm of the pair 1 + x.  fpmp2_accuracy::high: below 2^-7
      in magnitude log1p is the Taylor series x + x^2 P(x) in pair
      arithmetic instead, which avoids the cancellation in the buckets
      next to zero (there 1.8u^2 against 9.8u^2).
      Accuracy against MPFR: within 5u^2 for log, 6u^2 for log2 and
      log10, 10u^2 for log1p (8u^2 at high).
      Special cases: NaN->NaN, +-0->-inf, x<0->NaN, +inf->+inf;
      subnormal x is scaled into the normal range; log1p(-1) = -inf,
      log1p(x < -1) = NaN, log1p(+-0) = +-0.

    pow(x, y) for fp32mp2:
      pow(x, y) = exp(y * log|x|) with log|x| = e*ln2 + L from the shared
      log core (|L| <= 0.35).  The product y * log|x| is never rounded
      into a pair: an fp32mp2 t carries only 2^-48 * |t| of absolute
      accuracy, |t| reaches 88 before the result overflows, and every
      bit lost in t is lost in the result.  The exp kernel receives a
      directly reduced argument instead:
        r = (y*e - N1)*ln2 - N2*ln2 + y*L,   N = N1 + N2 = round(t/ln2)
      with N1 = round(y_hi*e): y*e as two exact products, y_hi*e - N1
      the exact fraction of y_hi*e, (y*e - N1)*ln2 a small pair times
      the ln2 pair, N2*ln2 from the three-piece ln2 of the exp
      reduction, y*L as an exact product with its exact cross terms,
      and the pieces combined by error-free sums wherever a piece is
      above 2^-48 * |y*L| (the products' low words are 2^-19 absolute
      at |y*L| = 35; adding them in float cost 2^-43 each).  The kernel
      itself (`__internal_fpmp2_exp_core` / `_exp_scales`) is exp's,
      factored out.  What remains is the log core's relative error
      times |y*L| <= 35 plus the kernel's own, in place of the old
      ~9 u^2 * |t|: 44 bits on the normal dataset (was 41), mean error
      3.4 u^2 (was 12.5) over [0.01, 100]^2.  Results below 2^-102
      (subnormal low limb) are limited by the format itself, as for
      exp.  Special cases (x <= 0, infinities, NaN, integer y) are
      unchanged.

    cbrt(x), rcbrt(x) for fp32mp2:
      x = r * 2^(3n), r in [1/4, 2) (denormals pre-scaled by 2^24).  Seed
      from the SFU lg2/ex2 pair (~2^-22), polished by one Newton step in
      float so that it sits within an ulp of the root; then one Halley
      step in which the residual (r - s^3, or 1 - r s^3) is an error-free
      pair and the correction is formed without a rounding of its own
      above 2^-47 (cbrt: pair denominator and a quotient corrected from
      its remainder; rcbrt: u s/3 as exact products with 1/3 as a pair).
      Every 2^-23-sized piece enters the high word of the correction
      through an error-free sum; left for the low word it costs 2^-47 in
      the final addition.  Host model against binary128: 1.5 / 1.4 u^2
      worst, independent of the seed.  Accuracy: 47 bits (cbrt was 45:
      its fp32mp2 Halley step rounded a 2^-22 correction through the
      1-ulp rcp.approx; rcbrt was 46).  The 2^n scaling is exact; the
      sign is restored at the end.

    hypot(x, y), rhypot(x, y) for fp32mp2:
      Both inputs scaled by one power of two chosen from the larger
      magnitude (exact; no overflow or underflow of the squares); x^2 +
      y^2 formed as a normalized pair from the two error-free squares
      with their cross terms 2 x_hi x_lo folded in by one fma each, the
      pieces joined by Fast2Sums in magnitude order; then the sqrt
      (rsqrt) core and the exact back-scaling by 2^k.  An infinite
      argument gives +inf (hypot) / +0 (rhypot) even when the other is
      NaN; rhypot(0, 0) = +inf.  Host model against binary128: 2.6 u^2
      worst, 0.3 mean (the sqrt core alone is 2.5).  Accuracy: 46 bits,
      against the 48 bits of the former double fallback, at a small
      fraction of its cost wherever FP64 is rationed.

    sinh(x), cosh(x), tanh(x) for fp32mp2:
      On the reduction and tables of exp and expm1: |x| is reduced to a
      multiple k of ln2/128 plus a pair phi, |phi| <= ln2/256, and none
      of the three evaluates a polynomial in x.
        sinh: e^(k ln2/128) = 2^n (1 + A) and its reciprocal
              2^n' (1 + B) from the table of expm1, combined with
              sinh(phi) and cosh(phi) without a division; for
              |x| < 0.34 the leading term is A - B, which keeps the
              relative accuracy down to x = 0.
        cosh: 2^(n-1) (y + 4^-n / y) with e^|x| = 2^n y from the tail
              of exp: one reciprocal.
        tanh: ((1 - z) + s) / ((1 + z) + s) with e^(2|x|) = 2^n (1 + s),
              z = 2^-n, below 3/4; 1 - 2 z / (y + z) above; and
              x - x^3/3 + 2 x^5/15 below 2^-12.
      sinh and cosh scale by 2^(n-1) at the end, so they stay finite up
      to their own overflow threshold ln(2^129) = 89.416, past the
      threshold 88.72 of exp.
      Accuracy against MPFR: within 6.2u^2 (sinh), 3.5u^2 (cosh) and
      7.4u^2 (tanh), u = 2^-24.
      fpmp2_accuracy::high types get a second kernel for sinh and tanh
      (the default and low levels share the one above): a polynomial
      in x next to zero (|x| < 1.25 for sinh, 0.655 for tanh) and
      (e - 1/e)/2 or 1 - 2/(e^(2|x|) + 1) with a refined division
      beyond, within 3.8u^2 (sinh) and 2.3u^2 (tanh), and slower past
      the polynomial's range.  cosh has one kernel for every level.
      Special cases: sinh(+-0) = +-0, sinh(+-inf) = +-inf,
      cosh(+-inf) = +inf, tanh(+-0) = +-0, tanh(+-inf) = +-1, NaN->NaN.

    asinh(x), acosh(x), atanh(x) for fp32mp2:
      Each forms the argument of one logarithm from the limbs, with
      every sum that can cancel taken exactly:
        asinh(|x|) = log1p(|x| + x^2/(1 + r)), r = sqrt(1 + x^2), |x| < 3/4
                   = log(|x| + r)                              up to 2^25
                   = log(|x|) + ln2                            beyond
        acosh(x)   = log1p(d + r), d = x - 1, r = sqrt(d (d + 2)), x < 5/4
                   = log(x + r)                                up to 2^25
                   = log(x) + ln2                              beyond
        atanh(|x|) = |x| + |x|^3 P(x^2)                        for |x| < 1/4
                   = log1p(2|x|/(1 - |x|))/2                   up to 1
      The log1p forms keep the relative accuracy of the result where it
      is small (x near 0, or near 1 for acosh); where the result is
      above 0.69 the logarithm of the sum itself is as accurate.  For
      x - 1 < 2^-60 acosh is sqrt(2 (x - 1)).
      Accuracy against MPFR: within 7u^2 (asinh), 10u^2 (acosh) and
      9.3u^2 (atanh).
      fpmp2_accuracy::high types get a second kernel for atanh, which
      splits the exact 2|x| off the argument of log1p and refines the
      division of the rest: within 5.2u^2.  asinh and acosh have one
      kernel for every level.
      Special cases: asinh(+-0) = +-0, asinh(+-inf) = +-inf,
      acosh(1) = +0, acosh(+inf) = +inf, acosh(x < 1) = NaN,
      atanh(+-0) = +-0, atanh(+-1) = +-inf, atanh(|x| > 1) = NaN,
      NaN->NaN.  The domains of acosh and atanh are decided on both
      limbs.

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
      ~27-bit results.)  A sparse pair - a tail more than 29 binades below
      the head, as arithmetic leaves it when a head lands on a power of
      two - carries more than 53 bits, and rounding it would move the
      remainder by the whole amount dropped; such an operand takes an
      exact reduction instead: the divisor and the running remainder on
      two 64-bit words in the frame of the finer operand (the divisor
      fits 126 bits there for tails down to 2^-102 ulp(hi); beyond that
      the 53-bit window applies), the dividend limb by limb in 64-bit
      arithmetic.  Dense pairs never take that path.  In that level every
      result is the nearest pair (head and tail each rounded to nearest
      even), so a result whose head has few bits keeps the tail bits the
      pair can hold.  fmod, remainder and remquo take the exact path in
      fpmp2_accuracy::high only; mid and low round the operands to the
      53-bit window and the result to 48 bits as before, which keeps
      their size (fmod 277 against 1087 SASS lines, 22 against 32
      registers) and speed (remainder 210 against 239 clk/ev) where they
      were - the exact block weighs on a caller's register budget even
      when it is never entered.
      remainder adds the round-to-
      nearest-even step (compare 2*ia vs My, ties broken by the parity of
      the accumulated quotient) and the |x| < |y| short-circuit.  Special
      cases: NaN or x = +-inf or y = 0 -> NaN; y = +-inf -> x.

    erf(x) for fp32mp2:
      erf(x) = -expm1(-|x|*P(|x|)).
      Core: degree-24 Remez polynomial P for the argument, followed by
      expm1 via argument reduction + polynomial, all in fp32mp2.

    erfc(x) for fp32mp2:
      |x| < 1/2: 1 - x Q(x^2), the small erf (no cancellation).
      |x| >= 1/2: erfc(x) = erfcx(|x|)*exp(-x^2).
      Core: erfcx(|x|) = s*R(s - 9/16) with s = 4/(|x|+4), from one reciprocal
      and a degree-18 minimax polynomial R, combined with a dedicated
      exp(-x^2) evaluation using a degree-11 polynomial, all in fp32mp2.
      Uses the identity erfc(-x) = 2 - erfc(x) for negative arguments.
      The result is 0 (or 2) for |x| > 10.0546875, where erfc(|x|) < 2^-150.
      fpmp2_accuracy::high: the erfcx kernel below in place of the
      one-reciprocal kernel for |x| >= 1/2 (about half its error, in more
      operations), with the same cut-off.  fp64mp2 has one erfc for every level.

    erfcx kernel (a >= 1/2), the basis of erfcx, normcdf and the inverse
    error functions, and of erfc at fpmp2_accuracy::high (the other levels
    of erfc have their own for |x| >= 1/2, above):
      (1 + 2a) e^{a^2} erfc(a) ~ P(t), t = (a - 4)/(a + 4), P of degree 22,
      both divisions refined with the exact remainder.

    erfcx(x) for fp32mp2:
      |x| < 1/2: exp(x^2) (1 - x Q(x^2)).  x >= 1/2: the kernel up to 40,
      then the asymptotic series (1/(x sqrt pi)) sum (-1)^k (2k-1)!!/(2x^2)^k,
      8 terms.  x <= -1/2: 2 exp(x^2) - erfcx(|x|); +inf below -9.3824.

    normcdf(x) for fp32mp2:
      y = |x|/sqrt 2 < 1/2: (1 +- y Q(y^2))/2; otherwise h = erfcx(y)
      exp(-x^2/2)/2 with the exact-argument exp applied to x itself, h for
      x < 0 and 1 - h for x > 0.  Zero below -14.5, one from 8.4 on.

    erfinv(x), erfcinv(q) for fp32mp2:
      Giles' single-precision erfinv as the seed (2^-22 up to
      w = -log(q(2 - q)) = 16, the Abramowitz-Stegun normal quantile 26.2.23
      below q = 1.1e-7), then one Halley step on erf(y) - x or erfc(y) - q
      with the small erf y Q(y^2) or the erfc tail and f' = -+ (2/sqrt pi)
      exp(-y^2) from the exact-argument exp.  erfinv(x) for |x| > 1/2 is erfcinv(1 - |x|)
      with 1 - |x| exact; erfcinv(q) for q in [1/2, 3/2] is erfinv(1 - q);
      q > 1 mirrors to -erfcinv(2 - q).  Below q = erfc(8) ~ 1e-29, where
      erfc's low limb is denormal, the equation ln erfcx(y) - y^2 = ln q is
      solved on the kernel instead.  erfinv(+-1) = +-inf, erfcinv(0) = +inf,
      erfcinv(2) = -inf, NaN outside, both limbs read.

    normcdfinv(p) for fp32mp2:
      Giles' polynomials with sqrt 2 folded into the coefficients, z = a R(w),
      a = 2p - 1, w = -log(4p(1-p)) from the fp32mp2 log.
      Central branch (w < 6.25): degree 22 in (w - 3.125).
      Tail 1 branch (6.25 <= w < 16): degree 18 in (sqrt w - 3.25).
      Tail 2 branch (w >= 16): degree 24 in (sqrt w - 7.25).
      Mixed Horner with 4 / 4 / 13 leading float coefficients, sized for a
      Gaussian sampler: 45-46 bits in the central region and tail 1, 41 bits
      in tail 2 (p < 5.6e-8), covering all float inputs including denormals.
      Returns +-infinity for p <= 0 or p >= 1, both limbs read: (1, -tiny) is
      inside the domain.  The icdf() wrappers clamp to +-FLT_MAX for safe
      Gaussian variate generation.

    boys_f0(x) for fp32mp2:
      Five ranges: degree 16 in (3 - x) below 4; degree 14 in (x - 5.65) and
      in (x - 9.38) up to 11.46; rsqrt(x) G(1/x) with G of degree 16 up to
      34.38; (sqrt pi / 2) rsqrt(x) beyond.  Mixed Horner (compensated for G),
      no erf, no sqrt.

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
      Argument reduction: x = n*(pi/2) + r, |r| <= 0.908.
      Three paths based on |x|:
        Tiny (|x| < pi/4):  no reduction needed.
        Fast (|x| < 2^20):  Cody-Waite with a 4-piece pi/2 (94 bits);
          x_hi - n*c_1 is one exact fma, n*c_2 and n*c_3 are split exactly,
          and the third-order terms are added as an exact triple, so r
          keeps 46 bits down to |r| ~ 2^-47 |x|.  |r| <= 0.908.
        Large (|x| >= 2^20): Payne-Hanek with a 256-bit 2/pi table and a
          96-bit fixed-point fraction (x_hi and x_lo combined before the
          pi/2 multiply), no fp64 operations; 46 bits down to |r| ~ 2^-46.
      Core: sin(r) = r (1 + r^2 S(r^2)) and cos(r) = 1 + r^2 C(r^2), S and C
            minimax polynomials of degree 6 on |r| <= 0.91 with three
            coefficients as float-float and four as floats; the rounding
            error of the single-precision part is carried into the ff
            Horner steps, which use Fast2Sum.
      sincos and tan evaluate both kernels; sin and cos evaluate one, its
      coefficients selected by the parity of n.
      Quadrant mapping via n mod 4 with sign/swap adjustment.
      sin(-0), tan(-0) and the sine of sincos(-0) are -0.
      fpmp2_accuracy::high: the fast path accumulates x_hi - n*c_1 (an exact
      product and a TwoSum) and the remaining terms in fp32mp2_high, and the
      core is one shared Taylor kernel, sin(r) via 8 terms (r through r^15)
      and cos(r) via 9 terms (1 through r^16) with r^2 shared, coefficients
      down to 1/9! (sin) and 1/10! (cos) as float-float; sin and cos call
      sincos.  About 1 u^2 less at the worst argument, in more operations.
      The tiny and Payne-Hanek paths, the quadrant mapping and the special
      values are those of the other levels.  fp64mp2 has one sin, cos, tan
      and sincos for every level.
      Accuracy: 46 bits (sin, cos), 45 bits (tan, the quotient of the two).

    sinpi(x), cospi(x), sincospi(x) for fp32mp2:
      The argument is already in units of pi, so reduction is an integer/fraction
      split rather than a radian Payne-Hanek or Cody-Waite reduction.
      x = n + f with |f| <= 1/2, then f is folded into [-1/4, 1/4] and multiplied
      by the 4-piece pi with exact products and error-free sums.  |x| < 2^-64 is
      evaluated as sinpi(x) = pi*x on a scaled argument, so denormal inputs keep
      their precision.  sin(r) and cos(r) reuse the sincos kernels: the
      Taylor kernel at fpmp2_accuracy::high, the minimax kernels otherwise.
      sinpi and cospi call sincospi. fp64mp2 stays on the double-precision
      fallback, the same at every level.
      Accuracy: 46 bits.

    asin(x), acos(x), atan(x), atan2(y, x) for fp32mp2:
      Two kernels behind one entry code, selected by the accuracy level of
      the type.  fpmp2_accuracy::low and the default level take the angle
      of a point (M, m), 0 <= m <= M, from a table of atan(i/16):
      atan(m/M) = atan(c) + atan(t), c = i/16, t = (m - c M)/(M + c m),
      |t| < 3/64, with a 4-coefficient kernel; asin and acos enter as the
      unit-circle point (x, sqrt((1 - |x|)(1 + |x|))), where the division
      becomes a product by a tabulated cosine.  At most 4 (atan) to 7.2
      (acos) u^2, 45 to 46 bits.
      fpmp2_accuracy::high evaluates 13-coefficient polynomials in
      fp32mp2 arithmetic: atan and atan2 after one fold and a refined
      division, asin and acos in x^2 below 0.575 and in 1 - |x| above.
      At most 1.9 (atan) to 3.4 (acos, atan2) u^2, 46 to 47 bits, for two
      to three times the instructions.
      The entry code is shared, so every level returns the C99 Annex F
      special values, keeps the sign of a zero argument (asin, atan,
      atan2), decides the domain of asin and acos on both limbs, returns
      NaN in both limbs, and scales the operands of atan2 by a common
      power of two.  fp64mp2 has one implementation for every level.

    Placeholder functions:
    - Delegate to standard double-precision system functions
    - Intended as API stubs; they do not provide full multi-precision accuracy yet

    fdim, remquo, lrint, lround, llrint, llround for fp32mp2:
      Exact on the limb pair, no FP64.  fdim compares the pairs
      lexicographically and takes the accurate difference (+0 when
      x <= y, NaN propagated, +inf when the difference overflows).
      remquo's remainder is remainder's, bit for bit at every level
      (same core, see above); its quotient is the low bits of
      rint(x / y) with the sign of x / y, modulo what the platform's
      own remquo uses (2^3 on the device as CUDA's, 2^31 on the host as
      glibc's), so that the fp32mp2 and the scalar / fp64mp2 results
      agree on each platform.  The four integer roundings convert the
      exact rint / round of the pair limb by limb; NaN, infinities and
      |x| >= 2^63 go to the platform's conversion of the high limb, as
      the scalar functions do.  The classification predicates,
      copysign, frexp, modf, ilogb and logb test the float limbs
      directly instead of widening them to double, and the predicates
      classify both limbs and combine the answers by logic alone (no
      arithmetic): a loose pair out of a low-accuracy operation can
      carry an infinity or a NaN in lo alone, or be (+inf, -inf), which
      is a NaN.  isnan: a NaN limb or two infinities of opposite sign;
      isinf: no NaN limb and one infinite limb or two of the same sign;
      isfinite: both limbs finite; signbit: hi's sign when the limbs
      agree or lo is zero (-0 is kept), else the sign of the larger
      limb, +0 when they cancel.

    fp64mp2 implementations:
    - When _CCCL_FPMP_FP128_MATH_FALLBACK=1, route through __float128 (libquadmath
      on host, CUDA fp128 intrinsics on device) for ~113-bit accuracy.
    - When _CCCL_FPMP_FP128_MATH_FALLBACK=0, fall back to double-precision math.
    - The fp64mp2 normcdfinv uses CUDA erfcinv on device, the fp32mp2 code on
      host (no standard erfcinv available).

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

#ifdef _CCCL_FPMP_USE_LIB
#  include <cuda/__fp/fpmp_math_impl_lib.h>
#endif // _CCCL_FPMP_USE_LIB

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
/*
 * ============================================================================
 * Freestanding API functions for fpmp2 class
 * ============================================================================
 *
 * Force-inlined like the fpmp2 operators. With the operators expanded, the larger functions look too
 * big to GCC to inline into the loops that call them, and those loops are then not vectorized.
 */

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> exp(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_exp(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> log(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_log(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> log2(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_log2(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> log10(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_log10(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> log1p(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_log1p(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_log1p(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
pow(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_pow(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> cbrt(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_cbrt(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> sin(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_sin(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_sin(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> cos(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_cos(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_cos(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE void
sincos(const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __s, fpmp2<_FpType, _TypeAcc>* __c) noexcept
{
  _FpType __sin_hi, __sin_lo, __cos_hi, __cos_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_sincos(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo);
  }
  else
  {
    ::cuda::__fpmp2_sincos(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo);
  }
  *__s = fpmp2<_FpType, _TypeAcc>(__sin_hi, __sin_lo);
  *__c = fpmp2<_FpType, _TypeAcc>(__cos_hi, __cos_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> asin(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_asin(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_asin(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> acos(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_acos(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_acos(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> atan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_atan(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_atan(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
atan2(const fpmp2<_FpType, _TypeAcc>& __y, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_atan2(__y.hi(), __y.lo(), __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_atan2(__y.hi(), __y.lo(), __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> sinh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_sinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_sinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> cosh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_cosh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> tanh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_tanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_tanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> erf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_erf(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> erfc(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_erfc(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_erfc(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> boys_f0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_boys_f0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> normcdfinv(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_normcdfinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// icdf takes no fpmp2 argument, so the accuracy tag of the result cannot be deduced
// and is the one place in this header that still defaults, like the class template.
template <fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<float, _TypeAcc> icdf(uint32_t __x) noexcept
{
  float __res_hi;
  float __res_lo;
  ::cuda::__fpmp2_icdf(__x, &__res_hi, &__res_lo);
  return fpmp2<float, _TypeAcc>(__res_hi, __res_lo);
}

template <fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<float, _TypeAcc> icdf(uint64_t __x) noexcept
{
  float __res_hi;
  float __res_lo;
  ::cuda::__fpmp2_icdf(__x, &__res_hi, &__res_lo);
  return fpmp2<float, _TypeAcc>(__res_hi, __res_lo);
}

// Inverse hyperbolic functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> acosh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_acosh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> asinh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_asinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> atanh(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_atanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_atanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Tangent
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> tan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_tan(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_tan(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Additional exponential/logarithmic functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> exp2(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_exp2(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> exp10(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_exp10(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> expm1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_expm1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_expm1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> logb(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_logb(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Rounding functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> ceil(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_ceil(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> floor(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_floor(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> trunc(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_trunc(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> round(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_round(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> rint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_rint(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> nearbyint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_nearbyint(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Absolute value
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> fabs(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_fabs(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Gamma functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> lgamma(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_lgamma(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> tgamma(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_tgamma(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Bessel functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> j0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_j0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> j1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_j1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> y0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_y0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> y1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_y1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
jn(int __n, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_jn(__n, __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
yn(int __n, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_yn(__n, __x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
cyl_bessel_i0(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_cyl_bessel_i0(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
cyl_bessel_i1(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_cyl_bessel_i1(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// CUDA-specific trigonometric functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> sinpi(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_sinpi(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_sinpi(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> cospi(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_cospi(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_cospi(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE void
sincospi(const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __s, fpmp2<_FpType, _TypeAcc>* __c) noexcept
{
  _FpType __sin_hi, __sin_lo, __cos_hi, __cos_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_sincospi(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo);
  }
  else
  {
    ::cuda::__fpmp2_sincospi(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo);
  }
  *__s = fpmp2<_FpType, _TypeAcc>(__sin_hi, __sin_lo);
  *__c = fpmp2<_FpType, _TypeAcc>(__cos_hi, __cos_lo);
}

// Normal distribution CDF and reciprocal functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> normcdf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_normcdf(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> rcbrt(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_rcbrt(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> erfcinv(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_erfcinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> erfinv(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_erfinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc> erfcx(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_erfcx(__x.hi(), __x.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
norm3d(const fpmp2<_FpType, _TypeAcc>& __a,
       const fpmp2<_FpType, _TypeAcc>& __b,
       const fpmp2<_FpType, _TypeAcc>& __c) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_norm3d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
norm4d(const fpmp2<_FpType, _TypeAcc>& __a,
       const fpmp2<_FpType, _TypeAcc>& __b,
       const fpmp2<_FpType, _TypeAcc>& __c,
       const fpmp2<_FpType, _TypeAcc>& __d) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_norm4d(
    __a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), __d.hi(), __d.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
rnorm3d(const fpmp2<_FpType, _TypeAcc>& __a,
        const fpmp2<_FpType, _TypeAcc>& __b,
        const fpmp2<_FpType, _TypeAcc>& __c) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_rnorm3d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
rnorm4d(const fpmp2<_FpType, _TypeAcc>& __a,
        const fpmp2<_FpType, _TypeAcc>& __b,
        const fpmp2<_FpType, _TypeAcc>& __c,
        const fpmp2<_FpType, _TypeAcc>& __d) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_rnorm4d(
    __a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), __d.hi(), __d.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Two-argument functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
fmax(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_fmax(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
fmin(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_fmin(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
max(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_max(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
min(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_min(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
fmod(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_fmod(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_fmod(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
remainder(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_remainder(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  }
  else
  {
    ::cuda::__fpmp2_remainder(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
hypot(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_hypot(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
copysign(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_copysign(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
fdim(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_fdim(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
nextafter(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_nextafter(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
rhypot(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_rhypot(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Functions with special signatures
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
remquo(const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y, int* __quo) noexcept
{
  _FpType __res_hi, __res_lo;
  if constexpr (_TypeAcc == fpmp2_accuracy::high)
  {
    ::cuda::__fpmp2_high_remquo(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo, __quo);
  }
  else
  {
    ::cuda::__fpmp2_remquo(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo, __quo);
  }
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
ldexp(const fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_ldexp(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
scalbn(const fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_scalbn(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
scalbln(const fpmp2<_FpType, _TypeAcc>& __x, long int __n) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_scalbln(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
frexp(const fpmp2<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{
  _FpType __res_hi, __res_lo;
  ::cuda::__fpmp2_frexp(__x.hi(), __x.lo(), &__res_hi, &__res_lo, __nptr);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE fpmp2<_FpType, _TypeAcc>
modf(const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __iptr) noexcept
{
  _FpType __res_hi, __res_lo, __i_hi, __i_lo;
  ::cuda::__fpmp2_modf(__x.hi(), __x.lo(), &__res_hi, &__res_lo, &__i_hi, &__i_lo);
  *__iptr = fpmp2<_FpType, _TypeAcc>(__i_hi, __i_lo);
  return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo);
}

// Functions returning integer types
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int ilogb(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_ilogb(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE long long int llrint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_llrint(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE long long int llround(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_llround(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE long int lrint(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_lrint(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE long int lround(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_lround(__x.hi(), __x.lo());
}

// Classification functions
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int fpmp_isfinite(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_isfinite(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int fpmp_isinf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_isinf(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int fpmp_isnan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_isnan(__x.hi(), __x.lo());
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int fpmp_signbit(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::__fpmp2_signbit(__x.hi(), __x.lo());
}

// The same tests under their standard names.
template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int isfinite(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_isfinite(__x);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int isinf(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_isinf(__x);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int isnan(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_isnan(__x);
}

template <typename _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int signbit(const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp_signbit(__x);
}

/*
 * Note: the fpmp2 warp-shuffle overloads (__shfl_sync, __shfl_xor_sync,
 * __shfl_down_sync, __shfl_up_sync) are thread-cooperation primitives, not math
 * functions, so they live in the core header <cuda/__fp/fpmp.h> (available via
 * <cuda/fpmp>) rather than here.
 */
_CCCL_END_NAMESPACE_CUDA

// ============================================================================
// cuda::std overloads for the standard <cmath> names.
//
// The emulated math lives in cuda, but a qualified
// cuda::std::<fn>(x) call suppresses ADL, so without these overloads it would
// silently narrow fpmp2 -> double (via the implicit conversion) and compute a
// native-double result. These forward to the cuda:: implementations
// (which unqualified / ADL calls already resolve to). Only names that cuda::std
// actually declares are provided; the CUDA-only extensions (rsqrt, exp10,
// rcbrt, sinpi, cospi, j0/j1/y0/y1, cyl_bessel_*, normcdf*, erf*inv, erfcx,
// norm*/rnorm*/rhypot, ...) have no cuda::std counterpart and are omitted.
// ============================================================================
_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _CCCL_FPMP_STD_UNARY(_Name)                                               \
  template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>                       \
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc> _Name( \
    const ::cuda::fpmp2<_FpType, _TypeAcc>& __x) noexcept                         \
  {                                                                               \
    return ::cuda::_Name(__x);                                                    \
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

#define _CCCL_FPMP_STD_BINARY(_Name)                                                                   \
  template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>                                            \
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc> _Name(                      \
    const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, const ::cuda::fpmp2<_FpType, _TypeAcc>& __y) noexcept \
  {                                                                                                    \
    return ::cuda::_Name(__x, __y);                                                                    \
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

#define _CCCL_FPMP_STD_UNARY_RET(_Ret, _Name)                                                              \
  template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>                                                \
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE _Ret _Name(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x) noexcept \
  {                                                                                                        \
    return ::cuda::_Name(__x);                                                                             \
  }

_CCCL_FPMP_STD_UNARY_RET(int, ilogb)
_CCCL_FPMP_STD_UNARY_RET(long long int, llrint)
_CCCL_FPMP_STD_UNARY_RET(long long int, llround)
_CCCL_FPMP_STD_UNARY_RET(long int, lrint)
_CCCL_FPMP_STD_UNARY_RET(long int, lround)

#undef _CCCL_FPMP_STD_UNARY_RET

// Functions with special signatures (extra scalar / out-pointer arguments).
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc>
ldexp(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return ::cuda::ldexp(__x, __n);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc>
scalbn(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return ::cuda::scalbn(__x, __n);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc>
scalbln(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, long int __n) noexcept
{
  return ::cuda::scalbln(__x, __n);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc>
frexp(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{
  return ::cuda::frexp(__x, __nptr);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc>
modf(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, ::cuda::fpmp2<_FpType, _TypeAcc>* __iptr) noexcept
{
  return ::cuda::modf(__x, __iptr);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE ::cuda::fpmp2<_FpType, _TypeAcc>
remquo(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x, const ::cuda::fpmp2<_FpType, _TypeAcc>& __y, int* __quo) noexcept
{
  return ::cuda::remquo(__x, __y, __quo);
}

// Classification functions.
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int isfinite(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::isfinite(__x);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int isinf(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::isinf(__x);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int isnan(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::isnan(__x);
}
template <class _FpType, ::cuda::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE int signbit(const ::cuda::fpmp2<_FpType, _TypeAcc>& __x) noexcept
{
  return ::cuda::signbit(__x);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_H
