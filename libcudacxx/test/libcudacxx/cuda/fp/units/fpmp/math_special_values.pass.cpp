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
#if defined(TEST_HAS_CUDA_FAST_MATH)
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
// through as well. The same templates serve fp32mp2 and fp64mp2.

template <typename MP2>
TEST_HOST_DEVICE_FUNC void test_rounding()
{
  using nl       = cuda::std::numeric_limits<MP2>;
  const MP2 pinf = nl::infinity();
  const MP2 qnan = nl::quiet_NaN();

#define CHECK_ROUNDING(name)                  \
  assert(is_pos_inf(name(pinf)));             \
  assert(is_neg_inf(name(-pinf)));            \
  assert(is_nan(name(qnan)));                 \
  assert(name(nl::max()) == nl::max());       \
  assert(name(nl::lowest()) == nl::lowest()); \
  assert(is_pos_zero(name(MP2(0.0f))));       \
  assert(is_neg_zero(name(MP2(-0.0f))));      \
  assert(is_neg_zero(name(MP2(-0.25f))) || name(MP2(-0.25f)) == MP2(-1.0f))

  CHECK_ROUNDING(ceil);
  CHECK_ROUNDING(floor);
  CHECK_ROUNDING(trunc);
  CHECK_ROUNDING(round);
  CHECK_ROUNDING(rint);
  CHECK_ROUNDING(nearbyint);

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
#if !defined(TEST_HAS_CUDA_FAST_MATH)
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
#if !defined(TEST_HAS_CUDA_FAST_MATH)
  assert(is_finite(normcdfinv(nl::denorm_min())) && normcdfinv(nl::denorm_min()) < T(0.0f));
#endif // !TEST_HAS_CUDA_FAST_MATH
}

// ---- boys_f0 -------------------------------------------------------------------
//
// F_0(0) = 1, F_0(+inf) = 0, F_0(NaN) = NaN. For a < 0 the function grows like
// exp(-a) / (2 |a|), which passes the largest float near a = -94: +inf from there
// down to -inf.

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

  assert(is_finite(boys_f0(T(-80.0f))) && boys_f0(T(-80.0f)) > T(0.0f));
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
  test_inverse_trig();
  test_trig();
  test_roots_and_distances();
  test_erf_family();
  test_modf_and_classify();
  return 0;
}
