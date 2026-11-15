// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 math at the special arguments of C99 Annex F.
//
//  Checks the class and the sign of the result -- NaN, +inf, -inf, zero -- for
//  the arguments outside the ordinary domain: zero, a negative argument, the
//  largest finite values, the infinities and NaN, and the few finite values
//  that sit on an edge (the overflow thresholds of the exponentials, the ends
//  of the inverse error functions, the quadrant boundaries of atan2, integers
//  and half-integers of the pi-scaled trig). Covered are the exponential and
//  logarithm families, the trigonometric and inverse trigonometric functions
//  with the Payne-Hanek range, the hyperbolic functions, cbrt, rcbrt, hypot,
//  rhypot, fdim, modf, the error and probability functions, boys_f0, the six
//  rounding functions (fp32mp2 and fp64mp2, whose zero result carries the sign
//  of the argument) and the classification predicates on loose pairs. Values
//  are compared against the double reference where one is in ISO <cmath>.
//
//  Left out on purpose:
//    - sqrt and rsqrt take no branch for special arguments and return NaN
//      for +inf, a negative argument, NaN and (on the device) a subnormal;
//      that is documented with the function list in
//      docs/libcudacxx/extended_api/fp/fpmp.rst.
//    - the sign of a zero result of the other functions, which fpmp2 does not
//      track; the same docs page has the note on the sign of zero.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

// ---- result classes ----------------------------------------------------------

template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_nan(MP2 r)
{
  return cuda::fpmp_isnan(r) != 0;
}

template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_pos_inf(MP2 r)
{
  return cuda::fpmp_isinf(r) != 0 && cuda::fpmp_signbit(r) == 0;
}

template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_neg_inf(MP2 r)
{
  return cuda::fpmp_isinf(r) != 0 && cuda::fpmp_signbit(r) != 0;
}

template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_finite(MP2 r)
{
  return cuda::fpmp_isfinite(r) != 0;
}

// Either sign: fpmp2 does not track the sign of zero in general.
template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_zero(MP2 r)
{
  return r.hi() == 0 && r.lo() == 0;
}

// The rounding functions and modf do carry the sign of the argument on a zero result.
template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_pos_zero(MP2 r)
{
  return is_zero(r) && cuda::fpmp_signbit(r) == 0;
}

template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_neg_zero(MP2 r)
{
  return is_zero(r) && cuda::fpmp_signbit(r) != 0;
}

template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_minus_one(MP2 r)
{
  return r.hi() == -1 && r.lo() == 0;
}

// For the finite limits (atan at +-inf is +-pi/2): well within fp32mp2 accuracy.
TEST_HOST_DEVICE_FUNC bool approx_eq(double a, double b)
{
  return cuda::std::fabs(a - b) <= 1e-9 * cuda::std::fabs(b);
}

// ---- log, log2, log10 --------------------------------------------------------
//
// log(+-0) = -inf, log(x < 0) = NaN including -inf, log(+inf) = +inf, log(NaN) = NaN.
// The largest and the smallest positive values stay finite.

TEST_HOST_DEVICE_FUNC void test_log()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  // assert per check rather than an accumulated flag, so a failure names the
  // function and the argument that broke.
#define CHECK_LOG(name)               \
  assert(is_neg_inf(name(T(0.0f))));  \
  assert(is_neg_inf(name(T(-0.0f)))); \
  assert(is_nan(name(T(-1.0f))));     \
  assert(is_nan(name(nl::lowest()))); \
  assert(is_nan(name(-pinf)));        \
  assert(is_pos_inf(name(pinf)));     \
  assert(is_nan(name(qnan)));         \
  assert(is_finite(name(nl::max()))); \
  assert(is_finite(name(nl::min()))); \
  assert(name(nl::max()) > T(0.0f));  \
  CHECK_LOG_SUBNORMAL(name)

  // A subnormal argument is finite and gives a negative result; --use_fast_math flushes it to
  // zero on the device (FTZ), so that part is left to the IEEE build.
#if TEST_HAS_CUDA_FAST_MATH
#  define CHECK_LOG_SUBNORMAL(name) static_cast<void>(0)
#else
#  define CHECK_LOG_SUBNORMAL(name)            \
    assert(is_finite(name(nl::denorm_min()))); \
    assert(name(nl::denorm_min()) < T(0.0f))
#endif // TEST_HAS_CUDA_FAST_MATH

  CHECK_LOG(log);
  CHECK_LOG(log2);
  CHECK_LOG(log10);

#undef CHECK_LOG
#undef CHECK_LOG_SUBNORMAL
}

// ---- atan --------------------------------------------------------------------
//
// atan(+-inf) = +-pi/2, atan(NaN) = NaN; the largest finite arguments round to +-pi/2.

TEST_HOST_DEVICE_FUNC void test_atan()
{
  using T              = cuda::fp32mp2;
  using nl             = cuda::std::numeric_limits<T>;
  const T pinf         = nl::infinity();
  const T qnan         = nl::quiet_NaN();
  const double half_pi = 1.5707963267948966;

  assert(is_nan(atan(qnan)));
  assert(is_finite(atan(pinf)) && approx_eq(static_cast<double>(atan(pinf)), half_pi));
  assert(is_finite(atan(-pinf)) && approx_eq(static_cast<double>(atan(-pinf)), -half_pi));
  assert(approx_eq(static_cast<double>(atan(nl::max())), half_pi));
  assert(approx_eq(static_cast<double>(atan(nl::lowest())), -half_pi));
}

// ---- ceil, floor, trunc, round, rint, nearbyint --------------------------------
//
// An infinity and NaN pass through unchanged, and a value beyond the last
// fractional magnitude is already an integer, so the largest finite values pass
// through as well. -0.25 goes to -1 under floor and to -0 under the other five
// (rint and nearbyint in the default rounding mode). The same templates serve
// fp32mp2 and fp64mp2.

template <typename MP2>
TEST_HOST_DEVICE_FUNC void test_rounding()
{
  using nl       = cuda::std::numeric_limits<MP2>;
  const MP2 pinf = nl::infinity();
  const MP2 qnan = nl::quiet_NaN();

#define CHECK_ROUNDING(name, at_minus_quarter) \
  assert(is_pos_inf(name(pinf)));              \
  assert(is_neg_inf(name(-pinf)));             \
  assert(is_nan(name(qnan)));                  \
  assert(name(nl::max()) == nl::max());        \
  assert(name(nl::lowest()) == nl::lowest());  \
  assert(is_pos_zero(name(MP2(0.0f))));        \
  assert(is_neg_zero(name(MP2(-0.0f))));       \
  assert(at_minus_quarter(name(MP2(-0.25f))))

  CHECK_ROUNDING(ceil, is_neg_zero);
  CHECK_ROUNDING(floor, is_minus_one);
  CHECK_ROUNDING(trunc, is_neg_zero);
  CHECK_ROUNDING(round, is_neg_zero);
  CHECK_ROUNDING(rint, is_neg_zero);
  CHECK_ROUNDING(nearbyint, is_neg_zero);

#undef CHECK_ROUNDING
}

// ---- normcdfinv ----------------------------------------------------------------
//
// Domain [0, 1]: normcdfinv(0) = -inf, normcdfinv(1) = +inf, normcdfinv(0.5) = 0.
// Outside the domain the result is NaN, also for a pair whose high limb sits
// exactly on an end and whose low limb points outward, and for +-inf and NaN.

TEST_HOST_DEVICE_FUNC void test_normcdfinv()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  assert(is_neg_inf(normcdfinv(T(0.0f))));
  assert(is_pos_inf(normcdfinv(T(1.0f))));
  assert(is_zero(normcdfinv(T(0.5f))));

  assert(is_nan(normcdfinv(T(-1.0f))));
  assert(is_nan(normcdfinv(T(2.0f))));
#if !TEST_HAS_CUDA_FAST_MATH
  // --use_fast_math flushes subnormals to zero on the device (FTZ), which turns -denorm_min
  // into -0 and the result into -inf; the subnormal arguments are left to the IEEE build.
  assert(is_nan(normcdfinv(-nl::denorm_min()))); // just below 0
#endif // !TEST_HAS_CUDA_FAST_MATH
  assert(is_nan(normcdfinv(T(1.0f, 0x1p-30f)))); // the pair just above 1
  assert(is_nan(normcdfinv(-pinf)));
  assert(is_nan(normcdfinv(pinf)));
  assert(is_nan(normcdfinv(qnan)));

  // The pairs just inside the ends are finite, large and of the right sign.
  assert(is_finite(normcdfinv(T(1.0f, -0x1p-30f))) && normcdfinv(T(1.0f, -0x1p-30f)) > T(0.0f));
#if !TEST_HAS_CUDA_FAST_MATH
  assert(is_finite(normcdfinv(nl::denorm_min())) && normcdfinv(nl::denorm_min()) < T(0.0f));
#endif // !TEST_HAS_CUDA_FAST_MATH
}

// ---- boys_f0 -------------------------------------------------------------------
//
// F_0(0) = 1, F_0(+inf) = 0, F_0(NaN) = NaN. The kernels cover a >= 0, the domain
// the function is used on; below 0 they extrapolate the polynomial for [0, 2) and
// lose its accuracy quickly, so no value is checked there. The function itself
// grows like exp(-a) / (2 |a|) for a < 0 and passes the largest float near
// a = -94: +inf from there down to -inf.

TEST_HOST_DEVICE_FUNC void test_boys_f0()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  assert(approx_eq(static_cast<double>(boys_f0(T(0.0f))), 1.0));
  assert(is_finite(boys_f0(nl::max())) && boys_f0(nl::max()) > T(0.0f));
  assert(is_zero(boys_f0(pinf)));
  assert(is_nan(boys_f0(qnan)));

  assert(is_pos_inf(boys_f0(T(-100.0f))));
  assert(is_pos_inf(boys_f0(nl::lowest())));
  assert(is_pos_inf(boys_f0(-pinf)));
}

// ---- exponentials at their edges ---------------------------------------------
//
// exp(+inf) = +inf, exp(-inf) = +0, exp(NaN) = NaN; the float at the overflow threshold,
// 0x1.62e430p+6 (88.72284), is above ln(FLT_MAX) and overflows to +inf rather than NaN,
// while the float below it is finite; exp2 overflows at 128, exp10 near 38.53.

TEST_HOST_DEVICE_FUNC void test_exp_edges()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

#define CHECK_EXP_FAMILY(name, below_zero) \
  assert(is_pos_inf(name(pinf)));          \
  assert(is_nan(name(qnan)));              \
  assert(name(-pinf) == T(below_zero))

  CHECK_EXP_FAMILY(exp, 0.0f);
  CHECK_EXP_FAMILY(exp2, 0.0f);
  CHECK_EXP_FAMILY(exp10, 0.0f);
  CHECK_EXP_FAMILY(expm1, -1.0f);
#undef CHECK_EXP_FAMILY

  assert(is_pos_inf(exp(T(0x1.62e430p+6f))));
  assert(is_pos_inf(exp(T(0x1.62e430p+6f, 1e-6f))));
  assert(is_finite(exp(T(0x1.62e42ep+6f))) && exp(T(0x1.62e42ep+6f)) > T(3.4e38f));
  assert(is_pos_inf(expm1(T(0x1.62e430p+6f))));
  assert(is_pos_inf(exp2(T(128.0f))));
  assert(is_finite(exp2(T(127.99f))));
  assert(is_pos_inf(exp10(T(38.6f))) && exp10(T(38.6f)).lo() == 0.0f);
  assert(is_finite(exp10(T(38.53f))));
  assert(is_zero(exp(T(-104.0f))) || exp(T(-104.0f)) > T(0.0f));
}

// ---- hyperbolic ----------------------------------------------------------------
//
// sinh and cosh stay finite up to |x| = 89.416 (e^|x| / 2 is taken directly from exp with
// its scale lowered by one binade), where exp alone overflows from 88.7228; tanh(+-inf) =
// +-1; asinh(+-inf) = +-inf, acosh(x < 1) = NaN, acosh(1) = 0, atanh(+-1) = +-inf, atanh(|x| > 1) = NaN.

TEST_HOST_DEVICE_FUNC void test_hyperbolic()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  assert(is_finite(sinh(T(89.0f))) && approx_eq(static_cast<double>(sinh(T(89.0f))), 2.2448064095871728e+38));
  assert(is_finite(cosh(T(89.0f))) && approx_eq(static_cast<double>(cosh(T(89.0f))), 2.2448064095871728e+38));
  assert(is_finite(sinh(T(-89.2f))) && sinh(T(-89.2f)) < T(-2.7e38f));
  assert(is_finite(cosh(T(89.4f))) && cosh(T(89.4f)) > T(3.3e38f));
  assert(is_finite(sinh(T(0x1.62e430p+6f, 1e-6f)))); // the exp threshold: e^x / 2 is finite
  assert(is_pos_inf(sinh(T(89.5f))) && is_neg_inf(sinh(T(-89.5f))));
  assert(is_pos_inf(cosh(T(89.5f))) && is_pos_inf(cosh(T(-89.5f))));
  assert(is_pos_inf(sinh(pinf)) && is_neg_inf(sinh(-pinf)) && is_nan(sinh(qnan)));
  assert(is_pos_inf(cosh(pinf)) && is_pos_inf(cosh(-pinf)) && is_nan(cosh(qnan)));
  assert(tanh(pinf) == T(1.0f) && tanh(-pinf) == T(-1.0f) && is_nan(tanh(qnan)));
  assert(is_pos_inf(asinh(pinf)) && is_neg_inf(asinh(-pinf)) && is_nan(asinh(qnan)));
  assert(is_pos_inf(acosh(pinf)) && is_nan(acosh(T(0.5f))) && is_zero(acosh(T(1.0f))) && is_nan(acosh(qnan)));
  assert(is_pos_inf(atanh(T(1.0f))) && is_neg_inf(atanh(T(-1.0f))) && is_nan(atanh(T(1.5f))) && is_nan(atanh(qnan)));
}

// At every accuracy level: expm1, log1p, sinh, tanh, asinh and atanh return a zero argument with
// its sign; a pair (128, lo) with lo < 0 is below the overflow threshold of exp2; and atanh of
// a pair (+-1, lo) with lo pointing inwards is finite down to a subnormal lo.

template <typename T>
TEST_HOST_DEVICE_FUNC void test_exp_hyperbolic_edges()
{
  assert(is_neg_zero(expm1(T(-0.0f))) && is_pos_zero(expm1(T(0.0f))));
  assert(is_neg_zero(log1p(T(-0.0f))) && is_pos_zero(log1p(T(0.0f))));
  assert(is_neg_zero(sinh(T(-0.0f))) && is_pos_zero(sinh(T(0.0f))));
  assert(is_neg_zero(tanh(T(-0.0f))) && is_pos_zero(tanh(T(0.0f))));
  assert(is_neg_zero(asinh(T(-0.0f))) && is_pos_zero(asinh(T(0.0f))));
  assert(is_neg_zero(atanh(T(-0.0f))) && is_pos_zero(atanh(T(0.0f))));

  assert(is_finite(exp2(T(128.0f, -1e-6f))) && exp2(T(128.0f, -1e-6f)) > T(3.4e38f));
  assert(is_pos_inf(exp2(T(128.0f))) && is_pos_inf(exp2(T(128.0f, 1e-6f))));

  // atanh(1 - d) = (ln 2 - ln d) / 2 to d / 4
  assert(approx_eq(static_cast<double>(atanh(T(1.0f, -0x1p-100f))), 35.00393261827724));
#if !TEST_HAS_CUDA_FAST_MATH
  assert(approx_eq(static_cast<double>(atanh(T(1.0f, -0x1p-140f))), 48.86687622947614));
  assert(approx_eq(static_cast<double>(atanh(T(-1.0f, 0x1p-140f))), -48.86687622947614));
#endif // !TEST_HAS_CUDA_FAST_MATH
}

// ---- inverse trigonometric and the quadrants of atan2 ---------------------------

TEST_HOST_DEVICE_FUNC void test_inverse_trig()
{
  using T         = cuda::fp32mp2;
  using nl        = cuda::std::numeric_limits<T>;
  const T pinf    = nl::infinity();
  const T qnan    = nl::quiet_NaN();
  const double pi = 3.141592653589793;

  assert(approx_eq(static_cast<double>(asin(T(1.0f))), pi / 2)
         && approx_eq(static_cast<double>(asin(T(-1.0f))), -pi / 2));
  assert(is_nan(asin(T(1.5f))) && is_nan(asin(T(-1.5f))) && is_nan(asin(qnan)) && is_nan(asin(pinf)));
  assert(is_zero(acos(T(1.0f))) && approx_eq(static_cast<double>(acos(T(-1.0f))), pi));
  assert(is_nan(acos(T(1.5f))) && is_nan(acos(qnan)) && is_nan(acos(-pinf)));
  // the branch beyond |x| = 0.575, against the double reference
  assert(approx_eq(static_cast<double>(asin(T(0.6f))), 0.6435011385956071));
  assert(approx_eq(static_cast<double>(acos(T(0.9f))), 0.45102686649323265));
  assert(approx_eq(static_cast<double>(asin(T(0.999f))), 1.5260715275839687));

  // atan2: the four quadrants, the axes and the infinities
  assert(approx_eq(static_cast<double>(atan2(T(1.0f), T(1.0f))), pi / 4));
  assert(approx_eq(static_cast<double>(atan2(T(1.0f), T(-1.0f))), 3 * pi / 4));
  assert(approx_eq(static_cast<double>(atan2(T(-1.0f), T(-1.0f))), -3 * pi / 4));
  assert(approx_eq(static_cast<double>(atan2(T(-1.0f), T(1.0f))), -pi / 4));
  assert(approx_eq(static_cast<double>(atan2(T(0.0f), T(-1.0f))), pi));
  assert(approx_eq(static_cast<double>(atan2(T(1.0f), T(0.0f))), pi / 2));
  assert(approx_eq(static_cast<double>(atan2(T(-1.0f), T(0.0f))), -pi / 2));
  assert(is_zero(atan2(T(0.0f), T(0.0f))) && is_zero(atan2(T(0.0f), T(1.0f))) && is_zero(atan2(T(1.0f), pinf)));
  assert(approx_eq(static_cast<double>(atan2(pinf, pinf)), pi / 4));
  assert(approx_eq(static_cast<double>(atan2(-pinf, -pinf)), -3 * pi / 4));
  assert(approx_eq(static_cast<double>(atan2(pinf, T(1.0f))), pi / 2));
  assert(approx_eq(static_cast<double>(atan2(T(1.0f), -pinf)), pi));
  assert(is_nan(atan2(qnan, T(1.0f))) && is_nan(atan2(T(1.0f), qnan)));
}

// ---- trigonometric: pi-scaled at integers and half-integers, the Payne-Hanek range ---------

TEST_HOST_DEVICE_FUNC void test_trig()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  assert(is_nan(sin(pinf)) && is_nan(cos(-pinf)) && is_nan(tan(pinf)) && is_nan(sin(qnan)) && is_nan(cos(qnan))
         && is_nan(tan(qnan)));
  assert(is_nan(sinpi(pinf)) && is_nan(cospi(pinf)) && is_nan(sinpi(qnan)) && is_nan(cospi(qnan)));

  // sinpi and cospi are exact at integers and half-integers
  assert(is_zero(sinpi(T(3.0f))) && is_zero(sinpi(T(-2.0f))) && is_zero(sinpi(T(1.0e6f))));
  assert(sinpi(T(0.5f)) == T(1.0f) && sinpi(T(1.5f)) == T(-1.0f) && sinpi(T(-0.5f)) == T(-1.0f));
  assert(is_zero(cospi(T(0.5f))) && is_zero(cospi(T(-1.5f))) && is_zero(cospi(T(2.5f))));
  assert(cospi(T(1.0f)) == T(-1.0f) && cospi(T(2.0f)) == T(1.0f) && cospi(T(-1.0f)) == T(-1.0f));

  // the Payne-Hanek range, |x| >= 2^20: the pair value is exact in double, so the double
  // reference is good to 1e-16
  assert(approx_eq(static_cast<double>(sin(T(0x1.4p21f))), 0.7460198878985261));
  assert(approx_eq(static_cast<double>(cos(T(0x1.4p21f))), 0.6659236644390034));
  assert(approx_eq(static_cast<double>(tan(T(0x1.4p21f))), 1.120278385852226));
  assert(approx_eq(static_cast<double>(sin(T(0x1.0p30f, 0x1.0p4f))), 0.3646929038840915));
  assert(approx_eq(static_cast<double>(cos(T(-0x1.8p25f))), -0.8961918292972255));
}

// sin, tan and the sine of sincos return a zero argument with its sign, at every accuracy level.

template <typename T>
TEST_HOST_DEVICE_FUNC void test_trig_zero()
{
  assert(is_neg_zero(sin(T(-0.0f))) && is_pos_zero(sin(T(0.0f))));
  assert(is_neg_zero(tan(T(-0.0f))) && is_pos_zero(tan(T(0.0f))));
  T s, c;
  sincos(T(-0.0f), &s, &c);
  assert(is_neg_zero(s) && c == T(1.0f));
  sincos(T(0.0f), &s, &c);
  assert(is_pos_zero(s) && c == T(1.0f));
  assert(cos(T(-0.0f)) == T(1.0f));
}

// sinpi, the sine of sincospi, erf and erfinv return a zero argument with its sign, sinpi of a
// nonzero integer has the sign of the argument, and cospi is +0 at every half-integer
// (IEEE 754-2019, 9.2), at every accuracy level.

template <typename T>
TEST_HOST_DEVICE_FUNC void test_pi_trig_and_erf_zero()
{
  assert(is_neg_zero(sinpi(T(-0.0f))) && is_pos_zero(sinpi(T(0.0f))));
  assert(is_pos_zero(sinpi(T(1.0f))) && is_neg_zero(sinpi(T(-1.0f))) && is_neg_zero(sinpi(T(-2.0f))));
  assert(is_pos_zero(cospi(T(0.5f))) && is_pos_zero(cospi(T(-0.5f))) && is_pos_zero(cospi(T(1.5f))));
  assert(is_pos_zero(cospi(T(-1.5f))) && is_pos_zero(cospi(T(2.5f))) && is_pos_zero(cospi(T(-2.5f))));
  T s, c;
  sincospi(T(-0.0f), &s, &c);
  assert(is_neg_zero(s) && c == T(1.0f));
  sincospi(T(0.0f), &s, &c);
  assert(is_pos_zero(s) && c == T(1.0f));
  sincospi(T(-0.5f), &s, &c);
  assert(s == T(-1.0f) && is_pos_zero(c));
  assert(is_neg_zero(erf(T(-0.0f))) && is_pos_zero(erf(T(0.0f))));
  assert(is_neg_zero(erfinv(T(-0.0f))) && is_pos_zero(erfinv(T(0.0f))));
}

// ---- roots, distances and the positive difference ------------------------------

TEST_HOST_DEVICE_FUNC void test_roots_and_distances()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  assert(is_pos_inf(cbrt(pinf)) && is_neg_inf(cbrt(-pinf)) && is_nan(cbrt(qnan)));
  assert(is_zero(cbrt(T(0.0f))) && is_zero(cbrt(T(-0.0f))) && cbrt(T(-8.0f)) == T(-2.0f));
  assert(is_zero(rcbrt(pinf)) && is_zero(rcbrt(-pinf)) && is_nan(rcbrt(qnan)));
  assert(is_pos_inf(rcbrt(T(0.0f))) && is_neg_inf(rcbrt(T(-0.0f))) && rcbrt(T(-8.0f)) == T(-0.5f));

  assert(is_pos_inf(hypot(pinf, qnan)) && is_pos_inf(hypot(qnan, -pinf)) && is_nan(hypot(qnan, T(1.0f))));
  assert(is_zero(hypot(T(0.0f), T(0.0f))) && hypot(T(-3.0f), T(4.0f)) == T(5.0f));
  assert(is_pos_inf(hypot(T(3e38f), T(3e38f))) && hypot(T(3e38f), T(3e38f)).lo() == 0.0f); // canonical (inf, 0)
  assert(is_zero(rhypot(pinf, qnan)) && is_zero(rhypot(-pinf, T(1.0f))) && is_nan(rhypot(qnan, T(1.0f))));
  assert(is_pos_inf(rhypot(T(0.0f), T(0.0f))) && approx_eq(static_cast<double>(rhypot(T(-3.0f), T(4.0f))), 0.2));

  // fdim: +0 when x <= y, NaN with a NaN, +inf when the difference overflows
  assert(is_zero(fdim(T(1.0f), T(2.0f))) && is_zero(fdim(T(1.0f), T(1.0f))) && fdim(T(2.0f), T(1.0f)) == T(1.0f));
  assert(is_nan(fdim(qnan, T(1.0f))) && is_nan(fdim(T(1.0f), qnan)));
  assert(is_pos_inf(fdim(pinf, T(1.0f))) && is_pos_inf(fdim(T(1.0f), -pinf)) && is_zero(fdim(-pinf, T(1.0f))));
  assert(is_pos_inf(fdim(nl::max(), nl::lowest())) && fdim(nl::max(), nl::lowest()).lo() == 0.0f);
}

// At every accuracy level: a NaN result of hypot, rhypot and the 3- and 4-operand norms is NaN
// in both limbs; an infinite operand of a norm gives +inf (+0 for the reciprocal) also next to
// a NaN, as for hypot; fmax and fmin order -0 below +0.

template <typename T>
TEST_HOST_DEVICE_FUNC bool is_nan_both(T r)
{
  return cuda::std::isnan(r.hi()) && cuda::std::isnan(r.lo());
}

template <typename T>
TEST_HOST_DEVICE_FUNC void test_norm_specials()
{
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();
  const T one  = T(1.0f);

  assert(is_nan_both(hypot(qnan, one)) && is_nan_both(rhypot(one, qnan)));
  assert(is_nan_both(norm3d(qnan, one, one)) && is_nan_both(norm4d(one, one, qnan, one)));
  assert(is_nan_both(rnorm3d(one, qnan, one)) && is_nan_both(rnorm4d(one, one, one, qnan)));
  assert(is_pos_inf(norm3d(pinf, qnan, one)) && is_pos_inf(norm3d(one, -pinf, one)));
  assert(is_pos_inf(norm4d(-pinf, qnan, one, one)) && is_pos_inf(norm4d(one, one, one, pinf)));
  assert(is_pos_zero(rnorm3d(pinf, qnan, one)) && is_pos_zero(rnorm3d(one, one, -pinf)));
  assert(is_pos_zero(rnorm4d(pinf, qnan, one, one)) && is_pos_zero(rnorm4d(one, -pinf, one, one)));
  assert(norm3d(T(2.0f), T(-3.0f), T(6.0f)) == T(7.0f) && norm4d(one, one, one, one) == T(2.0f));

  assert(is_pos_zero(fmax(T(-0.0f), T(0.0f))) && is_pos_zero(fmax(T(0.0f), T(-0.0f))));
  assert(is_neg_zero(fmin(T(-0.0f), T(0.0f))) && is_neg_zero(fmin(T(0.0f), T(-0.0f))));
}

// fp64mp2 fdim is taken on the limb pair: a difference in the low limbs alone is exact, and one
// that overflows is +inf.

TEST_HOST_DEVICE_FUNC void test_fdim_fp64mp2()
{
  using T  = cuda::fp64mp2;
  using nl = cuda::std::numeric_limits<T>;
  assert(fdim(T(1.0, 0x1p-70), T(1.0, 0x1p-72)) == T(0x1.8p-71));
  assert(is_zero(fdim(T(1.0, 0x1p-72), T(1.0, 0x1p-70))));
  assert(is_pos_inf(fdim(nl::max(), nl::lowest())) && fdim(nl::max(), nl::lowest()).lo() == 0.0);
  assert(is_nan(fdim(nl::quiet_NaN(), T(1.0))));
}

// ---- error and probability functions ------------------------------------------

TEST_HOST_DEVICE_FUNC void test_erf_family()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  assert(erf(pinf) == T(1.0f) && erf(-pinf) == T(-1.0f) && is_nan(erf(qnan)));
  assert(is_zero(erfc(pinf)) && erfc(-pinf) == T(2.0f) && is_nan(erfc(qnan)));
  assert(is_zero(erfcx(pinf)) && is_pos_inf(erfcx(-pinf)) && is_nan(erfcx(qnan)));
  // erfcx(x) ~ 2 exp(x^2) passes the largest fp32mp2 inside the low limb of x = -0x1.2c3cbcp+3
  assert(is_finite(erfcx(T(-0x1.2c3cbcp+3f))) && erfcx(T(-0x1.2c3cbcp+3f)) > T(0x1.ffffp+127f));
  assert(is_finite(erfcx(T(-0x1.2c3cbcp+3f, -0x1p-24f))));
  assert(is_pos_inf(erfcx(T(-0x1.2c3cbcp+3f, -0x1p-23f))) && erfcx(T(-0x1.2c3cbcp+3f, -0x1p-23f)).lo() == 0.0f);
  assert(is_pos_inf(erfcx(T(-0x1.2c3cbep+3f))) && is_pos_inf(erfcx(nl::lowest())));
#if !TEST_HAS_CUDA_FAST_MATH
  // 1 / (x sqrt(pi)) at the largest arguments is a subnormal number, which --use_fast_math
  // flushes to zero on the device (FTZ)
  assert(erfcx(T(0x1p127f)) > T(0x1p-128f) && erfcx(T(0x1p127f)) < T(0x1p-127f));
  assert(erfcx(nl::max()) > T(0x1p-129f) && erfcx(nl::max()) < T(0x1p-128f));
#endif // !TEST_HAS_CUDA_FAST_MATH
  assert(normcdf(pinf) == T(1.0f) && is_zero(normcdf(-pinf)) && is_nan(normcdf(qnan)));

  assert(is_pos_inf(erfinv(T(1.0f))) && is_neg_inf(erfinv(T(-1.0f))) && is_zero(erfinv(T(0.0f))));
  assert(is_nan(erfinv(T(1.5f))) && is_nan(erfinv(T(-1.5f))) && is_nan(erfinv(qnan)) && is_nan(erfinv(pinf)));
  assert(is_pos_inf(erfcinv(T(0.0f))) && is_neg_inf(erfcinv(T(2.0f))) && is_zero(erfcinv(T(1.0f))));
  assert(is_nan(erfcinv(T(-0.5f))) && is_nan(erfcinv(T(2.5f))) && is_nan(erfcinv(qnan)) && is_nan(erfcinv(pinf)));
  // the pairs just inside the ends are finite: (2, -2^-30) is a valid pair below 2
  assert(is_finite(erfcinv(T(2.0f, -0x1p-30f)))
         && approx_eq(static_cast<double>(erfcinv(T(2.0f, -0x1p-30f))), -4.328028275800946));
  assert(is_finite(erfinv(T(1.0f, -0x1p-30f)))
         && approx_eq(static_cast<double>(erfinv(T(1.0f, -0x1p-30f))), 4.328028275800946));
  assert(is_finite(erfcinv(T(0.0f, 0.0f) + T(0x1p-30f))) && erfcinv(T(0x1p-30f)) > T(4.0f));
}

// erfc at every accuracy level: beyond |x| = 10.0546875, where erfc(|x|) < 2^-150, the result is
// exactly 0 (or 2, with a zero low limb), and a NaN in the low limb alone gives NaN.

template <typename T>
TEST_HOST_DEVICE_FUNC void test_erfc_ends()
{
  using nl = cuda::std::numeric_limits<T>;
  assert(is_pos_zero(erfc(T(0x1.41c002p+3f))) && is_pos_zero(erfc(T(11.0f))) && is_pos_zero(erfc(T(27.0f))));
  assert(erfc(T(-0x1.41c002p+3f)).hi() == 2.0f && erfc(T(-0x1.41c002p+3f)).lo() == 0.0f);
  assert(erfc(T(-27.0f)).hi() == 2.0f && erfc(T(-27.0f)).lo() == 0.0f);
#if !TEST_HAS_CUDA_FAST_MATH
  assert(erfc(T(10.0f)) > T(0.0f)); // subnormal, just below the cut-off
#endif // !TEST_HAS_CUDA_FAST_MATH
  assert(is_nan(erfc(T(1.0f, nl::quiet_NaN().hi()))) && is_nan(erfc(T(0.25f, nl::quiet_NaN().hi()))));
}

// ---- a zero high limb over a nonzero low limb -----------------------------------
//
// Low-accuracy arithmetic can leave one: (1, 2^-30) - 1 is (0, 2^-30) and (2^24, 0.25) - 2^24
// is (0, 0.25). Such an argument is not a zero, and the functions that return a zero argument
// with its sign, or unchanged, must not take it for one.

// The fp32mp2 kernels give a pair with one nonzero limb the accuracy of that limb.
TEST_HOST_DEVICE_FUNC bool roughly_eq(double a, double b)
{
  return cuda::std::fabs(a - b) <= 1e-6 * cuda::std::fabs(b);
}

TEST_HOST_DEVICE_FUNC void test_loose_zero_high_limb()
{
  using T = cuda::fp32mp2_low;
  const T x(0.0f, 0x1p-30f);
  assert(approx_eq(static_cast<double>(sin(x)), 0x1p-30));
  assert(approx_eq(static_cast<double>(tan(x)), 0x1p-30));
  assert(approx_eq(static_cast<double>(sinpi(x)), 3.141592653589793 * 0x1p-30));
  assert(approx_eq(static_cast<double>(erf(x)), 1.1283791670955126 * 0x1p-30));
  assert(approx_eq(static_cast<double>(sin(-x)), -0x1p-30));
  assert(approx_eq(static_cast<double>(sinpi(-x)), -3.141592653589793 * 0x1p-30));

  const T q(0.0f, 0.25f);
  assert(roughly_eq(static_cast<double>(expm1(q)), 0.2840254166877415));
  assert(roughly_eq(static_cast<double>(log1p(q)), 0.22314355131420976));
  assert(roughly_eq(static_cast<double>(sinh(q)), 0.2526123168081683));
  assert(roughly_eq(static_cast<double>(tanh(q)), 0.24491866240370913));
  assert(roughly_eq(static_cast<double>(expm1(-q)), -0.22119921692859512));
  assert(roughly_eq(static_cast<double>(tanh(-q)), -0.24491866240370913));

  using D = cuda::fp64mp2_low;
  const D w(0.0, 0.25);
  assert(approx_eq(static_cast<double>(atan(w)), 0.24497866312686414));
  assert(approx_eq(static_cast<double>(asin(w)), 0.25268025514207865));
  assert(approx_eq(static_cast<double>(atan2(w, D(1.0))), 0.24497866312686414));
  assert(approx_eq(static_cast<double>(atan2(w, D(-1.0))), 2.8966139904629289));
  assert(approx_eq(static_cast<double>(atan2(-w, D(-1.0))), -2.8966139904629289));
}

// ---- llrint and llround at the ends of long long --------------------------------
//
// -2^63 + lo and 2^63 - |lo| are in range although their high limb is not below 2^63, and
// so is a pair below 2^63 whose rounding carries into the high limb.

TEST_HOST_DEVICE_FUNC void test_llrint_ends()
{
  using T               = cuda::fp32mp2;
  const long long llmax = (cuda::std::numeric_limits<long long>::max)();
  const long long llmin = (cuda::std::numeric_limits<long long>::min)();
  const long long p38   = 1LL << 38;

  assert(llrint(T(-0x1p63f)) == llmin && llround(T(-0x1p63f)) == llmin);
  assert(llrint(T(-0x1p63f, 1.0f)) == llmin + 1 && llround(T(-0x1p63f, 1.0f)) == llmin + 1);
  assert(llrint(T(-0x1p63f, 0x1p38f)) == llmin + p38 && llround(T(-0x1p63f, 0x1p38f)) == llmin + p38);
  assert(llrint(T(0x1p63f, -1.0f)) == llmax && llround(T(0x1p63f, -1.0f)) == llmax);
  assert(llrint(T(0x1p63f, -0x1p38f)) == llmax - p38 + 1 && llround(T(0x1p63f, -0x1p38f)) == llmax - p38 + 1);
  // ties: to even for llrint, away from zero for llround
  assert(llrint(T(-0x1p63f, 0.5f)) == llmin && llround(T(-0x1p63f, 0.5f)) == llmin);
  assert(llrint(T(-0x1p63f, 1.5f)) == llmin + 2 && llround(T(-0x1p63f, 1.5f)) == llmin + 1);
  assert(llrint(T(0x1p63f, -1.5f)) == llmax - 1 && llround(T(0x1p63f, -1.5f)) == llmax);
  // the rounded pair is renormalized to (2^63, -2^38)
  assert(llrint(T(0x1.fffffep62f, 0x1p38f)) == llmax - p38 + 1);
  assert(llround(T(0x1.fffffep62f, 0x1p38f)) == llmax - p38 + 1);
  assert(llrint(T(-0x1.fffffep62f, -0x1p38f)) == llmin + p38);
}

// ---- modf and the classification predicates on loose pairs --------------------

TEST_HOST_DEVICE_FUNC void test_modf_and_classify()
{
  using T      = cuda::fp32mp2;
  using nl     = cuda::std::numeric_limits<T>;
  const T pinf = nl::infinity();
  const T qnan = nl::quiet_NaN();

  T ip;
  T fr = modf(pinf, &ip);
  assert(is_pos_zero(fr) && is_pos_inf(ip));
  fr = modf(-pinf, &ip);
  assert(is_neg_zero(fr) && is_neg_inf(ip));
  fr = modf(qnan, &ip);
  assert(is_nan(fr) && is_nan(ip));
  fr = modf(T(-2.5f), &ip);
  assert(fr == T(-0.5f) && ip == T(-2.0f));
  fr = modf(T(-3.0f), &ip);
  assert(is_neg_zero(fr) && ip == T(-3.0f));

  // a loose pair (low limb outside the renormalized contract) classifies by both limbs
  assert(cuda::fpmp_isnan(T(1.0f, nl::quiet_NaN().hi())) != 0);
  assert(cuda::fpmp_isinf(T(1.0f, nl::infinity().hi())) != 0);
  assert(cuda::fpmp_isfinite(T(1.0f, nl::infinity().hi())) == 0);
  assert(cuda::fpmp_isfinite(T(1.0f, 2.0f)) != 0);
  // and the integer conversions route such a pair to the scalar fallback instead of UB
  (void) llrint(T(1.0f, nl::quiet_NaN().hi()));
  (void) llround(T(1.0f, nl::infinity().hi()));

  // A head of exactly +-2^63 is in range when the tail points inward, with the tail rounded
  // as part of the pair (rint ties to even, round ties away from zero).
  assert(llrint(T(-0x1p63f, 1.0f)) == LLONG_MIN + 1);
  assert(llround(T(-0x1p63f, 1.0f)) == LLONG_MIN + 1);
  assert(llrint(T(-0x1p63f, 0.0f)) == LLONG_MIN);
  assert(llrint(T(0x1p63f, -1.0f)) == LLONG_MAX);
  assert(llround(T(0x1p63f, -1.0f)) == LLONG_MAX);
  assert(llrint(T(0x1p63f, -0x1p39f)) == LLONG_MAX - 0x7FFFFFFFFFLL); // 2^63 - 2^39, the head itself
  assert(llrint(T(0x1p63f, -2.5f)) == LLONG_MAX - 1); // ties to even: 2^63 - 2
  assert(llround(T(0x1p63f, -2.5f)) == LLONG_MAX - 1); // ties away from zero: 2^63 - 2
  assert(llrint(T(0x1p63f, -1.5f)) == LLONG_MAX - 1); // ties to even: 2^63 - 2
  assert(llround(T(0x1p63f, -1.5f)) == LLONG_MAX); // ties away from zero: 2^63 - 1
  assert(llrint(T(-0x1p63f, 0.5f)) == LLONG_MIN); // ties to even: -2^63
  assert(llround(T(-0x1p63f, 0.5f)) == LLONG_MIN); // ties away from zero: -2^63
  assert(llrint(T(-0x1p63f, 1.5f)) == LLONG_MIN + 2);
  assert(llround(T(-0x1p63f, 1.5f)) == LLONG_MIN + 1);
  // out of range either way: the scalar function's value on the head
  assert(llrint(T(0x1p63f, 0.0f)) == ::cuda::std::llrint(0x1p63));
  assert(llround(T(-0x1p63f, -1.0f)) == ::cuda::std::llround(-0x1p63));
}

int main(int, char**)
{
  // force_include.h makes this main __host__ __device__ and runs it twice, on the
  // host and then in a kernel, so every check covers both.
  test_log();
  test_atan();
  test_rounding<cuda::fp32mp2>();
  test_rounding<cuda::fp64mp2>();
  test_normcdfinv();
  test_boys_f0();
  test_exp_edges();
  test_hyperbolic();
  test_exp_hyperbolic_edges<cuda::fp32mp2>();
  test_exp_hyperbolic_edges<cuda::fp32mp2_low>();
  test_exp_hyperbolic_edges<cuda::fp32mp2_high>();
  test_inverse_trig();
  test_trig();
  test_trig_zero<cuda::fp32mp2>();
  test_trig_zero<cuda::fp32mp2_low>();
  test_trig_zero<cuda::fp32mp2_high>();
  test_pi_trig_and_erf_zero<cuda::fp32mp2>();
  test_pi_trig_and_erf_zero<cuda::fp32mp2_low>();
  test_pi_trig_and_erf_zero<cuda::fp32mp2_high>();
  test_roots_and_distances();
  test_norm_specials<cuda::fp32mp2>();
  test_norm_specials<cuda::fp32mp2_low>();
  test_norm_specials<cuda::fp32mp2_high>();
  test_fdim_fp64mp2();
  test_erf_family();
  test_erfc_ends<cuda::fp32mp2>();
  test_erfc_ends<cuda::fp32mp2_low>();
  test_erfc_ends<cuda::fp32mp2_high>();
  test_loose_zero_high_limb();
  test_llrint_ends();
  test_modf_and_classify();
  return 0;
}
