// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

// fp32mp2 sinh, cosh, tanh, asinh, acosh and atanh: within a few u^2 (u = 2^-24) of an MPFR
// reference over the whole range of each, including small arguments, the arguments where a
// function changes path, sinh and cosh between the overflow thresholds of exp and their own,
// and acosh and atanh next to 1; the results that must be exact; the C99 Annex F special
// values; and the domains of acosh and atanh decided on both limbs.
//
// sinh, tanh and atanh have two fp32mp2 kernels: fp32mp2 and fp32mp2_low get the fast one, and
// fp32mp2_high one with a smaller worst-case error. Every check runs on all three types, the
// accuracy check against the bound of the kernel the type gets.

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

enum hyp_fn
{
  fn_sinh,
  fn_cosh,
  fn_tanh,
  fn_asinh,
  fn_acosh,
  fn_atanh,
  fn_count
};

template <class T>
TEST_HOST_DEVICE_FUNC T call(hyp_fn fn, T x)
{
  switch (fn)
  {
    case fn_sinh:
      return cuda::sinh(x);
    case fn_cosh:
      return cuda::cosh(x);
    case fn_tanh:
      return cuda::tanh(x);
    case fn_asinh:
      return cuda::asinh(x);
    case fn_acosh:
      return cuda::acosh(x);
    default:
      return cuda::atanh(x);
  }
}

struct hyp_case
{
  hyp_fn fn;
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
template <class T>
TEST_HOST_DEVICE_FUNC double rel_err_u2(T r, hyp_case c)
{
  double d = ((static_cast<double>(r.hi()) - c.ref_hi) + static_cast<double>(r.lo())) - c.ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(c.ref_hi) * 0x1p48;
}

// The largest error MPFR sweeps of 2M to 4M arguments per range found, rounded up. Default and low
// level: 6.2 for sinh, 3.5 for cosh, 7.4 for tanh, 6.9 for asinh, 9.7 for acosh and 9.3 for atanh.
// High level: 3.8 for sinh, 2.3 for tanh and 5.2 for atanh; cosh, asinh and acosh have one kernel.
template <class T>
TEST_HOST_DEVICE_FUNC void test_accuracy(bool high)
{
  const double tolerance_default[fn_count] = {7.5, 4.5, 9.0, 8.0, 11.0, 10.5};
  const double tolerance_high[fn_count]    = {4.5, 4.5, 3.0, 8.0, 11.0, 6.5};
  const double* tolerance                  = high ? tolerance_high : tolerance_default;
  // MPFR 4.2.2, 256-bit precision: the function, x, and its value. Small arguments, the worst
  // arguments of the sweeps, the arguments where a function changes path, and the ends of each
  // range.
  const hyp_case cases[] = {
    {fn_sinh, 0x1p-20f, 0x1.8p-46f, 0x1.00000060002abp-20, -0x1.55495554c84cdp-74},
    {fn_sinh, -0x1.69933cp-9f, -0x1.e86b0cp-34f, -0x1.69935b02137fp-9, 0x1.0e98ab16d7582p-63},
    {fn_sinh, 0x1.5c28f6p-2f, 0x1.12e0bep-30f, 0x1.62e821d2479e9p-2, -0x1.09949201ebd32p-60},
    {fn_sinh, -0x1.fdd2f2p-2f, 0x1.f7100ap-27f, -0x1.099314be4e3b8p-1, 0x1.32882b3176352p-57},
    {fn_sinh, 0x1.4p+1f, -0x1.01b2b2p-25f, 0x1.83368c1582809p+2, -0x1.53ee5d7c9874p-52},
    {fn_sinh, 0x1.0c9976p+3f, 0x0p+0f, 0x1.14348b4eae196p+11, 0x1.a544177164e14p-43},
    {fn_sinh, -0x1.0c948cp+3f, -0x1.04c726p-24f, -0x1.140a255780eap+11, 0x1.a28bb4fe7020ep-43},
    {fn_sinh, 0x1.44p+4f, 0x1.ad7f2ap-22f, 0x1.290d7e9b12fedp+28, 0x1.1161cc6ecbc8dp-26},
    {fn_sinh, 0x1.0d33cap+6f, -0x1.539e2ap-20f, 0x1.1146244382c31p+96, 0x1.889305844535ep+41},
    {fn_sinh, 0x1.63p+6f, 0x1.0c6f7ap-20f, 0x1.070c7a21b2ed7p+127, -0x1.ff4ac5aaa5c3cp+73},
    {fn_sinh, 0x1.65999ap+6f, -0x1.0c6f7ap-19f, 0x1.f7e13e3e06012p+127, 0x1.59caac17fdac7p+73},
    {fn_sinh, -0x1.65a3d8p+6f, 0x0p+0f, -0x1.fcf22f89549efp+127, 0x1.416ab84e3261bp+71},
    {fn_cosh, 0x1p-10f, 0x0p+0f, 0x1.00000800000abp+0, -0x1.5553e93e93814p-54},
    {fn_cosh, 0x1.c46f3cp-6f, 0x1.25a28p-32f, 0x1.0018fd2ffa4adp+0, 0x1.58e59f2dc228cp-54},
    {fn_cosh, 0x1p+0f, 0x0p+0f, 0x1.8b07551d9f55p+0, 0x1.30af4a040065bp-54},
    {fn_cosh, -0x1.cac826p+1f, 0x0p+0f, 0x1.206d844b273bdp+4, 0x1.705dc35eacef1p-50},
    {fn_cosh, 0x1.9714dep+2f, -0x1.acb3dcp-24f, 0x1.214fc598bccbep+8, -0x1.a1981e51b256p-47},
    {fn_cosh, 0x1.44p+4f, 0x1.ad7f2ap-22f, 0x1.290d7e9b12fedp+28, 0x1.2cf5a9d3e19d1p-26},
    {fn_cosh, 0x1.3c533p+6f, 0x1.9d3364p-19f, 0x1.108013f7b52aep+113, 0x1.893f6cd8bfcap+59},
    {fn_cosh, 0x1.63p+6f, 0x1.0c6f7ap-20f, 0x1.070c7a21b2ed7p+127, -0x1.ff4ac5aaa5c3cp+73},
    {fn_cosh, -0x1.65999ap+6f, 0x1.0c6f7ap-19f, 0x1.f7e13e3e06012p+127, 0x1.59caac17fdac7p+73},
    {fn_tanh, 0x1p-20f, 0x1.8p-46f, 0x1.0000005fffaabp-20, -0x1.556d554cd5cccp-74},
    {fn_tanh, 0x1.04d1c2p-16f, -0x1.5dc54cp-41f, 0x1.04d1c150c31b9p-16, -0x1.777793a609e48p-70},
    {fn_tanh, 0x1.fffffep-13f, 0x1.c25c26p-44f, 0x1.fffffd58da0fep-13, 0x1.0a6ef2ff2490dp-68},
    {fn_tanh, 0x1.0f7e5cp-12f, -0x1.80c2eep-37f, 0x1.0f7e5ad9d5f3dp-12, -0x1.a20b8011b691bp-66},
    {fn_tanh, 0x1.020ddcp-4f, -0x1.db12eap-31f, 0x1.01b697d03b135p-4, -0x1.5445590f4e711p-58},
    {fn_tanh, 0x1p-2f, 0x1.12e0bep-30f, 0x1.f597ea89ee3d9p-3, 0x1.44b2f05304c4ap-58},
    {fn_tanh, -0x1.1b9d82p-1f, -0x1.c03a96p-27f, -0x1.01c616242d472p-1, 0x1.aa382a81061d8p-55},
    {fn_tanh, 0x1.7ffffep-1f, 0x1.12e0bep-30f, 0x1.45323d28db57dp-1, -0x1.ce191ba942f7p-55},
    {fn_tanh, 0x1.8p-1f, 0x0p+0f, 0x1.45323e552f228p-1, 0x1.39d5832bf78fbp-56},
    {fn_tanh, -0x1.4p+1f, 0x1.01b2b2p-25f, -0x1.f9258259cccd5p-1, -0x1.2aeaaa7ab1cb1p-55},
    {fn_tanh, 0x1.1578p+3f, 0x1.f3e538p-23f, 0x1.fffffe06ac27dp-1, -0x1.03b7c5909f49ap-55},
    {fn_tanh, 0x1.14p+4f, -0x1.0c6f7ap-21f, 0x1.fffffffffffedp-1, 0x1.17e3a07efd05fp-55},
    {fn_asinh, 0x1p-20f, 0x1.8p-46f, 0x1.0000005fffd55p-20, 0x1.5549555a1da23p-74},
    {fn_asinh, 0x1.1f2af4p-5f, 0x1.d3b888p-30f, 0x1.1f1be8a670985p-5, 0x1.5afa63c5b3035p-60},
    {fn_asinh, -0x1.139d86p-1f, -0x1.96b926p-26f, -0x1.07ca0a5bce622p-1, -0x1.8ac71353064cap-58},
    {fn_asinh, 0x1.7ffffep-1f, 0x1.12e0bep-30f, 0x1.62e42e5ce93cfp-1, -0x1.0ba71868679f1p-56},
    {fn_asinh, 0x1.8p-1f, 0x0p+0f, 0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56},
    {fn_asinh, -0x1.32c856p+0f, -0x1.97cap-26f, -0x1.03d2604cc602ep+0, -0x1.dc613c5dc5db7p-54},
    {fn_asinh, 0x1.f75eaep+4f, 0x1.0c6f7ap-21f, 0x1.0918c229a8e9fp+2, -0x1.c33eca61f69f1p-54},
    {fn_asinh, -0x1.a676f8p+22f, -0x1.66bf8ep-3f, -0x1.0717d0155074dp+4, -0x1.a825589d54b1p-50},
    {fn_asinh, 0x1.fffffep+24f, 0x1p+0f, 0x1.205966eab4f12p+4, 0x1.2dca04964b171p-50},
    {fn_asinh, 0x1.03664p+25f, -0x1p+0f, 0x1.208f6f478a0cap+4, 0x1.daaa169a4d257p-50},
    {fn_asinh, -0x1.c105acp+93f, 0x1.b1ae4ep+68f, -0x1.06def48ae0c78p+6, 0x1.c8de0f77ddabbp-50},
    {fn_asinh, 0x1.c363ccp+127f, 0x0p+0f, 0x1.6528f3ed1f546p+6, 0x1.34ca6cb2fa927p-48},
    {fn_acosh, 0x1p+0f, 0x1.4484cp-100f, 0x1.979e8af2d9e33p-50, -0x1.4887c9558c8e6p-104},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_acosh, 0x1p+0f, 0x1p-149f, 0x1p-74, -0x1.5555555555555p-227},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_acosh, 0x1.00009p+0f, 0x1.08173cp-25f, 0x1.1003c1e02345cp-8, -0x1.1186fdde3446p-62},
    {fn_acosh, 0x1.10b41ep+0f, 0x1.19119p-28f, 0x1.6fee485e34c62p-2, 0x1.f7b6ec9d411e3p-56},
    {fn_acosh, 0x1.214a1cp+0f, -0x1.1b0a1ap-25f, 0x1.025b73e4ba11bp-1, 0x1.e0aeb41b52becp-55},
    {fn_acosh, 0x1.3ffffep+0f, 0x1.12e0bep-30f, 0x1.62e42aa5c245ep-1, 0x1.4a5cd47c45e61p-57},
    {fn_acosh, 0x1.4p+0f, 0x0p+0f, 0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56},
    {fn_acosh, 0x1.4de87ep+0f, 0x1.9839d8p-25f, 0x1.85f3776df8a48p-1, -0x1.6195fa55eb12p-57},
    {fn_acosh, 0x1.1a3404p+5f, 0x1.e34ce4p-22f, 0x1.10647786bb04fp+2, 0x1.214a307122b6ap-52},
    {fn_acosh, 0x1.16d8aap+23f, -0x1.7c1f1p-2f, 0x1.0b8947ccf804ep+4, -0x1.b9fe0f2b58e4ap-50},
    {fn_acosh, 0x1.fffffep+24f, 0x1p+0f, 0x1.205966eab4f12p+4, 0x1.5b94082c962d5p-51},
    {fn_acosh, 0x1.03664p+25f, -0x1p+0f, 0x1.208f6f478a0cap+4, 0x1.5dff4de54e11cp-50},
    {fn_acosh, 0x1.10db32p+50f, 0x1.c9c38p+24f, 0x1.1b506ecb1e475p+5, 0x1.95ff71ef502b8p-56},
    {fn_acosh, 0x1.c363ccp+127f, 0x0p+0f, 0x1.6528f3ed1f546p+6, 0x1.34ca6cb2fa927p-48},
    {fn_atanh, 0x1p-20f, 0x1.8p-46f, 0x1.0000006000555p-20, 0x1.556d55622b224p-74},
    {fn_atanh, 0x1.fffffep-3f, 0x1.12e0bep-30f, 0x1.058aeea95373fp-2, 0x1.c5c5c03059dap-56},
    {fn_atanh, 0x1.000cbp-2f, 0x1.fa9914p-27f, 0x1.0598794a3dc14p-2, -0x1.0ec165e912646p-56},
    {fn_atanh, -0x1.046a8p-2f, -0x1.e77a04p-27f, -0x1.0a423424a8dc2p-2, 0x1.f0557c23c577ap-56},
    {fn_atanh, 0x1p-1f, 0x0p+0f, 0x1.193ea7aad030bp-1, -0x1.a256f99caabebp-55},
    {fn_atanh, -0x1.8b2154p-1f, 0x0p+0f, -0x1.064d046926b8cp+0, 0x1.00739b32837f7p-54},
    {fn_atanh, 0x1.f26618p-1f, -0x1.2d720ep-26f, 0x1.13b55238a2a11p+1, -0x1.dc6db0ed8a208p-53},
    {fn_atanh, 0x1.fffffep-1f, 0x1.ad7f2ap-26f, 0x1.1df57403ceb02p+3, 0x1.977ad5c7b2fd5p-52},
    {fn_atanh, 0x1p+0f, -0x1.b7cdfep-34f, 0x1.7b8104260ed23p+3, -0x1.704dd0c08c1a8p-51},
    {fn_atanh, 0x1p+0f, -0x1.4484cp-100f, 0x1.1715325f48c8fp+5, 0x1.5c62747f6235p-49},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_atanh, -0x1p+0f, 0x1.16c2p-133f, -0x1.732fac73bfc14p+5, 0x1.a03bd492c3fabp-49},
    {fn_atanh, 0x1p+0f, -0x1p-149f, 0x1.9fe3682cd3be4p+5, 0x1.4aa84b6b3b125p-49},
#endif // !TEST_HAS_CUDA_FAST_MATH
  };
  for (hyp_case c : cases)
  {
    const T r = call(c.fn, T(c.x_hi, c.x_lo));
    assert(rel_err_u2(r, c) <= tolerance[c.fn]);
  }
}

template <class F>
struct exact_case
{
  hyp_fn fn;
  F x_hi;
  F x_lo;
  F r; // expected high limb, compared bit for bit (NaN matches any NaN)
};

template <class F>
TEST_HOST_DEVICE_FUNC bool same(F x, F y)
{
  if (cuda::std::isnan(y))
  {
    return cuda::std::isnan(x);
  }
  return x == y && cuda::std::signbit(x) == cuda::std::signbit(y);
}

// The high limb is the expected one. The low limb is zero if `exact_pair` or if the result is
// zero, infinite or subnormal, and NaN if the result is NaN.
template <class T, class F, int N>
TEST_HOST_DEVICE_FUNC void check_exact(const exact_case<F> (&cases)[N], bool exact_pair)
{
  for (const exact_case<F>& c : cases)
  {
    const T r = call(c.fn, T(c.x_hi, c.x_lo));
    assert(same(r.hi(), c.r));
    if (cuda::std::isnan(c.r))
    {
      assert(cuda::std::isnan(r.lo()));
    }
    else if (exact_pair || !cuda::std::isnormal(c.r))
    {
      assert(r.lo() == F(0));
    }
  }
}

// Exact results, with a zero low limb: cosh(0) = 1, acosh(1) = +0, tanh from 17.5 on, and the odd
// functions of an argument below 2^-25, which is the result itself. (The high-level atanh adds
// the x^3/3 term to its low limb there.)
template <class T>
TEST_HOST_DEVICE_FUNC void test_exact(bool high)
{
  const exact_case<float> cases[] = {
    {fn_cosh, 0.0f, 0.0f, 1.0f},
    {fn_cosh, -0.0f, 0.0f, 1.0f},
    {fn_cosh, 0x1p-30f, 0.0f, 1.0f},
    {fn_acosh, 1.0f, 0.0f, 0.0f},
    {fn_tanh, 17.5f, 0.0f, 1.0f},
    {fn_tanh, -20.0f, 0.0f, -1.0f},
    {fn_sinh, 0x1p-26f, 0.0f, 0x1p-26f},
    {fn_tanh, -0x1p-26f, 0.0f, -0x1p-26f},
    {fn_asinh, 0x1p-30f, 0.0f, 0x1p-30f},
  };
  check_exact<T>(cases, true);
  if (!high)
  {
    const exact_case<float> atanh_cases[] = {
      {fn_atanh, -0x1p-40f, 0.0f, -0x1p-40f},
    };
    check_exact<T>(atanh_cases, true);
  }
}

template <class T, class F>
TEST_HOST_DEVICE_FUNC void test_special_values()
{
  const F inf  = cuda::std::numeric_limits<F>::infinity();
  const F nan  = cuda::std::numeric_limits<F>::quiet_NaN();
  const F maxn = cuda::std::numeric_limits<F>::max();
  const F minn = cuda::std::numeric_limits<F>::min();
#if TEST_HAS_CUDA_FAST_MATH
  // --use_fast_math flushes subnormals to zero on the device (FTZ): the smallest normal stands
  // in for the smallest subnormal there
  const F mind = minn;
#else
  const F mind = cuda::std::numeric_limits<F>::denorm_min();
#endif // TEST_HAS_CUDA_FAST_MATH
  const exact_case<F> cases[] = {
    // sinh: x itself for a zero or tiny x, +-inf at +-inf and past the range
    {fn_sinh, nan, F(0), nan},
    {fn_sinh, -nan, F(0), nan},
    {fn_sinh, inf, F(0), inf},
    {fn_sinh, -inf, F(0), -inf},
    {fn_sinh, maxn, F(0), inf},
    {fn_sinh, -maxn, F(0), -inf},
    {fn_sinh, F(0), F(0), F(0)},
    {fn_sinh, -F(0), F(0), -F(0)},
    {fn_sinh, mind, F(0), mind},
    {fn_sinh, -mind, F(0), -mind},
    {fn_sinh, -minn, F(0), -minn},
    // cosh: 1 for a zero or tiny x, +inf at +-inf and past the range
    {fn_cosh, nan, F(0), nan},
    {fn_cosh, -nan, F(0), nan},
    {fn_cosh, inf, F(0), inf},
    {fn_cosh, -inf, F(0), inf},
    {fn_cosh, maxn, F(0), inf},
    {fn_cosh, -maxn, F(0), inf},
    {fn_cosh, F(0), F(0), F(1)},
    {fn_cosh, -F(0), F(0), F(1)},
    {fn_cosh, mind, F(0), F(1)},
    {fn_cosh, -minn, F(0), F(1)},
    // tanh: x itself for a zero or tiny x, +-1 at +-inf
    {fn_tanh, nan, F(0), nan},
    {fn_tanh, -nan, F(0), nan},
    {fn_tanh, inf, F(0), F(1)},
    {fn_tanh, -inf, F(0), F(-1)},
    {fn_tanh, maxn, F(0), F(1)},
    {fn_tanh, -maxn, F(0), F(-1)},
    {fn_tanh, F(0), F(0), F(0)},
    {fn_tanh, -F(0), F(0), -F(0)},
    {fn_tanh, mind, F(0), mind},
    {fn_tanh, -mind, F(0), -mind},
    {fn_tanh, -minn, F(0), -minn},
    // asinh: x itself for a zero or tiny x, +-inf at +-inf
    {fn_asinh, nan, F(0), nan},
    {fn_asinh, -nan, F(0), nan},
    {fn_asinh, inf, F(0), inf},
    {fn_asinh, -inf, F(0), -inf},
    {fn_asinh, F(0), F(0), F(0)},
    {fn_asinh, -F(0), F(0), -F(0)},
    {fn_asinh, mind, F(0), mind},
    {fn_asinh, -mind, F(0), -mind},
    {fn_asinh, -minn, F(0), -minn},
    // acosh: +0 at 1, +inf at +inf, NaN below 1
    {fn_acosh, nan, F(0), nan},
    {fn_acosh, -nan, F(0), nan},
    {fn_acosh, inf, F(0), inf},
    {fn_acosh, -inf, F(0), nan},
    {fn_acosh, F(1), F(0), F(0)},
    {fn_acosh, F(0.5), F(0), nan},
    {fn_acosh, F(0), F(0), nan},
    {fn_acosh, -F(0), F(0), nan},
    {fn_acosh, mind, F(0), nan},
    {fn_acosh, F(-1), F(0), nan},
    {fn_acosh, F(-2), F(0), nan},
    {fn_acosh, -maxn, F(0), nan},
    // atanh: x itself for a zero or tiny x, +-inf at +-1, NaN beyond
    {fn_atanh, nan, F(0), nan},
    {fn_atanh, -nan, F(0), nan},
    {fn_atanh, F(1), F(0), inf},
    {fn_atanh, F(-1), F(0), -inf},
    {fn_atanh, F(1.5), F(0), nan},
    {fn_atanh, F(-2), F(0), nan},
    {fn_atanh, inf, F(0), nan},
    {fn_atanh, -inf, F(0), nan},
    {fn_atanh, maxn, F(0), nan},
    {fn_atanh, F(0), F(0), F(0)},
    {fn_atanh, -F(0), F(0), -F(0)},
    {fn_atanh, mind, F(0), mind},
    {fn_atanh, -mind, F(0), -mind},
    {fn_atanh, -minn, F(0), -minn},
    // acosh and atanh of a high limb of 1: the low limb decides the domain
    {fn_acosh, F(1), -mind, nan},
    {fn_acosh, F(1), -minn, nan},
    {fn_atanh, F(1), mind, nan},
    {fn_atanh, F(-1), -minn, nan},
  };
  check_exact<T>(cases, false);
}

// Results at the ends of the fp32mp2 range. sinh and cosh are finite past ln(2^128) = 88.72,
// where exp overflows, up to ln(2^129) = 89.416; tanh rounds to 1 before 17.5; acosh and atanh
// of a high limb of 1 depend on a subnormal low limb; and asinh and acosh are finite at FLT_MAX.
template <class T>
TEST_HOST_DEVICE_FUNC void test_range_ends()
{
  const float inf                 = cuda::std::numeric_limits<float>::infinity();
  const exact_case<float> cases[] = {
    {fn_sinh, 0x1.62e43p+6f, 0.0f, 0x1.000004p+127f},
    {fn_sinh, 89.0f, 0.0f, 0x1.51c2ccp+127f},
    {fn_sinh, 0x1.65a9f8p+6f, 0.0f, 0x1.ffffd8p+127f}, // the float below ln(2^129)
    {fn_sinh, 0x1.65a9fap+6f, 0.0f, inf}, // and the one above
    {fn_sinh, -89.0f, 0.0f, -0x1.51c2ccp+127f},
    {fn_sinh, 89.5f, 0.0f, inf},
    {fn_sinh, -1000.0f, 0.0f, -inf},
    {fn_cosh, 89.0f, 0.0f, 0x1.51c2ccp+127f},
    {fn_cosh, 0x1.65a9f8p+6f, 0.0f, 0x1.ffffd8p+127f},
    {fn_cosh, 0x1.65a9fap+6f, 0.0f, inf},
    {fn_cosh, -89.0f, 0.0f, 0x1.51c2ccp+127f},
    {fn_cosh, -1000.0f, 0.0f, inf},
    {fn_tanh, 17.25f, 0.0f, 1.0f},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_acosh, 1.0f, 0x1p-149f, 0x1p-74f},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_acosh, 0x1.fffffep+127f, 0.0f, 0x1.65a9f8p+6f},
    {fn_asinh, -0x1.fffffep+127f, 0.0f, -0x1.65a9f8p+6f},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_atanh, 1.0f, -0x1p-149f, 0x1.9fe368p+5f},
    {fn_atanh, -1.0f, 0x1p-149f, -0x1.9fe368p+5f},
#endif // !TEST_HAS_CUDA_FAST_MATH
  };
  check_exact<T>(cases, false);
}

int main(int, char**)
{
  test_accuracy<cuda::fp32mp2>(false);
  test_accuracy<cuda::fp32mp2_low>(false);
  test_accuracy<cuda::fp32mp2_high>(true);
  test_exact<cuda::fp32mp2>(false);
  test_exact<cuda::fp32mp2_low>(false);
  test_exact<cuda::fp32mp2_high>(true);
  test_special_values<cuda::fp32mp2, float>();
  test_special_values<cuda::fp32mp2_low, float>();
  test_special_values<cuda::fp32mp2_high, float>();
  test_range_ends<cuda::fp32mp2>();
  test_range_ends<cuda::fp32mp2_low>();
  test_range_ends<cuda::fp32mp2_high>();

  return 0;
}
