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
//  largest finite values, the infinities and NaN. Covered are log, log2 and
//  log10, atan, the six rounding functions, normcdfinv and boys_f0. The
//  rounding functions are checked for fp32mp2 and fp64mp2 alike, the rest for
//  fp32mp2, which has native pair kernels for them.
//
//  Left out on purpose:
//    - sqrt and rsqrt take no branch for special arguments and return NaN
//      for +inf, a negative argument, NaN and (on the device) a subnormal;
//      that is documented with the function list in
//      docs/libcudacxx/extended_api/fp/fpmp.rst.
//    - the sign of a zero result, which fpmp2 does not track; the same docs page
//      has the note on the sign of zero, next to the one on sqrt and rsqrt.
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

// Either sign: fpmp2 does not track the sign of zero.
template <typename MP2>
TEST_HOST_DEVICE_FUNC bool is_zero(MP2 r)
{
  return r.hi() == 0 && r.lo() == 0;
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
#define CHECK_LOG(name)                      \
  assert(is_neg_inf(name(T(0.0f))));         \
  assert(is_neg_inf(name(T(-0.0f))));        \
  assert(is_nan(name(T(-1.0f))));            \
  assert(is_nan(name(nl::lowest())));        \
  assert(is_nan(name(-pinf)));               \
  assert(is_pos_inf(name(pinf)));            \
  assert(is_nan(name(qnan)));                \
  assert(is_finite(name(nl::max())));        \
  assert(is_finite(name(nl::min())));        \
  assert(is_finite(name(nl::denorm_min()))); \
  assert(name(nl::max()) > T(0.0f));         \
  assert(name(nl::denorm_min()) < T(0.0f))

  CHECK_LOG(log);
  CHECK_LOG(log2);
  CHECK_LOG(log10);

#undef CHECK_LOG
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
  assert(is_zero(name(MP2(0.0f))));           \
  assert(is_zero(name(MP2(-0.0f))))

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
  assert(is_nan(normcdfinv(-nl::denorm_min()))); // just below 0
  assert(is_nan(normcdfinv(T(1.0f, 0x1p-30f)))); // the pair just above 1
  assert(is_nan(normcdfinv(-pinf)));
  assert(is_nan(normcdfinv(pinf)));
  assert(is_nan(normcdfinv(qnan)));

  // The pairs just inside the ends are finite, large and of the right sign.
  assert(is_finite(normcdfinv(T(1.0f, -0x1p-30f))) && normcdfinv(T(1.0f, -0x1p-30f)) > T(0.0f));
  assert(is_finite(normcdfinv(nl::denorm_min())) && normcdfinv(nl::denorm_min()) < T(0.0f));
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
  return 0;
}
