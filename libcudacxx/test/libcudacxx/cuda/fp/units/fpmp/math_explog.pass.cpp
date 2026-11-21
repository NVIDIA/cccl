// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

// fp32mp2 exp, exp2, exp10, expm1, log, log2, log10 and log1p: within a few u^2 (u = 2^-24) of
// an MPFR reference over the whole range of each, including large |x| for the exponentials
// (reducing x to the format's own precision erred by up to 180u^2 on these cases) and
// arguments near 1 and near 0 for the logarithms; the results that must be exact; the C99
// Annex F special values; and gradual underflow below FLT_MIN. The special values are checked
// for fp64mp2 as well. fp32mp2_high evaluates expm1 and log1p below 2^-7 by their Taylor series,
// which is checked against the worst arguments of the other levels there.

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

enum explog_fn
{
  fn_exp,
  fn_exp2,
  fn_exp10,
  fn_expm1,
  fn_log,
  fn_log2,
  fn_log10,
  fn_log1p,
  fn_count
};

template <class T>
TEST_HOST_DEVICE_FUNC T call(explog_fn fn, T x)
{
  switch (fn)
  {
    case fn_exp:
      return cuda::exp(x);
    case fn_exp2:
      return cuda::exp2(x);
    case fn_exp10:
      return cuda::exp10(x);
    case fn_expm1:
      return cuda::expm1(x);
    case fn_log:
      return cuda::log(x);
    case fn_log2:
      return cuda::log2(x);
    case fn_log10:
      return cuda::log10(x);
    default:
      return cuda::log1p(x);
  }
}

struct explog_case
{
  explog_fn fn;
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
template <class T>
TEST_HOST_DEVICE_FUNC double rel_err_u2(T r, explog_case c)
{
  double d = ((static_cast<double>(r.hi()) - c.ref_hi) + static_cast<double>(r.lo())) - c.ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(c.ref_hi) * 0x1p48;
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_accuracy()
{
  // The largest error MPFR sweeps of 2M-4M arguments per range found, rounded up: 3.2 for exp,
  // exp2 and exp10, 4.5 for expm1, 4.7 for log, 5.3 for log2 and log10, 5.7 for log1p.
  const double tolerance[fn_count] = {4.0, 4.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0};
  // MPFR 4.2.2, 256-bit precision: the function, x, and its value. Small arguments, the worst
  // arguments of the sweeps, and the ends of each range with a normal low limb.
  const explog_case cases[] = {
    {fn_exp, 0x1p-40f, 0x1.8p-66f, 0x1.0000000001p+0, 0x1.80020000018p-66},
    {fn_exp, 0x1.333334p-2f, -0x1.2e5d9ep-30f, 0x1.599059077ded5p+0, 0x1.6800ff54c5432p-57},
    {fn_exp, -0x1.54488p-2f, -0x1.37173p-28f, 0x1.6f3d8dfea60b9p-1, -0x1.79c8fa47f53d4p-55},
    {fn_exp, 0x1p-1f, 0x0p+0f, 0x1.a61298e1e069cp+0, -0x1.b4690082a4906p-55},
    {fn_exp, 0x1.21ac76p+2f, 0x1.fe0fdcp-23f, 0x1.719be993167aep+6, -0x1.6dde39d4942b6p-48},
    {fn_exp, -0x1.756f5ep+4f, -0x1.ea9dfcp-21f, 0x1.41565aa77a173p-34, -0x1.cc856f03a5da6p-88},
    {fn_exp, 0x1.216962p+6f, 0x1.f10a68p-19f, 0x1.4de1c2c4ebef2p+104, -0x1.062c5a9dbd0dcp+49},
    {fn_exp, 0x1.62p+6f, -0x1.4f8b58p-19f, 0x1.99b943f44bd72p+127, 0x1.c02bd0af0ef5dp+73},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp, -0x1.04p+6f, 0x1.a0133p-19f, 0x1.2b2bca9807b6ep-94, -0x1.4ce7560ef3be3p-152},
    {fn_exp, -0x1.15p+6f, 0x0p+0f, 0x1.111d9e41aab28p-100, 0x1.42a9b2ffab22cp-154},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_exp2, 0x1p-40f, 0x1.8p-66f, 0x1.0000000000b17p+0, 0x1.0c1d2e6afb0b5p-55},
    {fn_exp2, -0x1.e6c42ep-2f, 0x1.feaeb2p-27f, 0x1.704692388023cp-1, 0x1.e4e6bb5a50858p-59},
    {fn_exp2, 0x1.20ba2ap+2f, 0x1.df06c6p-23f, 0x1.6ce6c07da485p+4, 0x1.59374db0d7959p-50},
    {fn_exp2, 0x1.5p+3f, 0x0p+0f, 0x1.6a09e667f3bcdp+10, -0x1.bdd3413b26456p-44},
    {fn_exp2, -0x1.4ba2c2p+5f, -0x1.f1ad8p-20f, 0x1.75a5433c4cd34p-42, 0x1.37700ef2ce3ffp-97},
    {fn_exp2, 0x1.c63f42p+6f, -0x1.f195c2p-19f, 0x1.79e06b0dc4a88p+113, 0x1.f7ee4d74516a9p+58},
    {fn_exp2, 0x1.fep+6f, 0x1.d5c316p-19f, 0x1.6a0a1ff796a8dp+127, -0x1.382b4c1b155eap+69},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp2, -0x1.69p+6f, 0x1.27476cp-20f, 0x1.ae8a0f1920baap-91, -0x1.eed818b358c1ep-149},
    {fn_exp2, -0x1.8ep+6f, 0x0p+0f, 0x1.6a09e667f3bcdp-100, -0x1.bdd3413b26456p-154},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_exp10, 0x1p-40f, 0x1.8p-66f, 0x1.00000000024d7p+0, 0x1.8e14ee67885dap-54},
    {fn_exp10, -0x1.2c0474p-3f, 0x1.c9e902p-28f, 0x1.6d683e2d71224p-1, 0x1.da18cd01b542fp-57},
    {fn_exp10, -0x1.f4c5a4p+0f, 0x1.1a4752p-25f, 0x1.6a8041a8cbcaep-7, 0x1.604b1b81a4d61p-61},
    {fn_exp10, 0x1.e1b34p+1f, 0x0p+0f, 0x1.6a60f99f56cafp+12, -0x1.526f14ac23008p-42},
    {fn_exp10, 0x1.6p+2f, 0x1.172c42p-23f, 0x1.34d0f7154d8e3p+18, 0x1.e0d54cbcd5196p-36},
    {fn_exp10, 0x1.330d44p+5f, -0x1.762aacp-20f, 0x1.6a2a3d7aca435p+127, -0x1.3d406a641c349p+73},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp10, -0x1.94p+4f, 0x1.d87248p-22f, 0x1.16754dd710017p-84, -0x1.18a544068832ap-138},
    {fn_exp10, -0x1.ep+4f, 0x0p+0f, 0x1.4484bfeebc2ap-100, -0x1.e72f6d3e432b6p-154},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_expm1, 0x1p-40f, 0x1.8p-66f, 0x1.00000060008p-40, 0x1.8000aaf2ab6abp-106},
    {fn_expm1, -0x1.8p-51f, 0x1p-77f, -0x1.7fffffbfffffep-51, 0x1.fffff9ffffff6p-106},
    {fn_expm1, 0x1.70fde8p-9f, 0x1.dd7c2p-34f, 0x1.7182ffa3f9bf2p-9, -0x1.68008db1ac3d8p-63},
    {fn_expm1, -0x1.661644p-9f, -0x1.8073p-38f, -0x1.659928a62eacfp-9, 0x1.433df17921326p-65},
    {fn_expm1, 0x1.032c32p-4f, 0x1.f8b532p-29f, 0x1.0b8c4218c34afp-4, -0x1.939825f3921acp-58},
    {fn_expm1, 0x1.00050cp-3f, -0x1.d00246p-28f, 0x1.10b5d9dd80cd1p-3, 0x1.7bcbc48d2e4a8p-57},
    {fn_expm1, -0x1.333334p-2f, 0x1.12e0bep-30f, -0x1.0966f3528e8e1p-2, 0x1.87e4e9dcbd581p-56},
    {fn_expm1, 0x1.622b58p-2f, -0x1.c21076p-28f, 0x1.a722481882995p-2, -0x1.0a22f67e421cp-60},
    {fn_expm1, -0x1.666666p-2f, 0x0p+0f, -0x1.2e663e8af3182p-2, -0x1.8dc88e6a1d64bp-56},
    {fn_expm1, 0x1.21ac76p+2f, 0x1.fe0fdcp-23f, 0x1.6d9be993167aep+6, -0x1.6dde39d4942b6p-48},
    {fn_expm1, -0x1.4p+2f, 0x1.6d127ep-23f, -0x1.fc8cd7fa2775ep-1, -0x1.e7360a2636a3fp-55},
    {fn_expm1, 0x1.7b2d54p+4f, 0x1.fb085cp-21f, 0x1.23ff05035baaap+34, 0x1.cc0fcb18db87ep-22},
    {fn_expm1, -0x1.ep+4f, 0x0p+0f, -0x1.ffffffffffcb5p-1, -0x1.1f3d53808b2afp-56},
    {fn_expm1, 0x1.9p+5f, 0x1.421f6p-20f, 0x1.1910545b0f9d8p+72, 0x1.be54fdce16cebp+17},
    {fn_expm1, 0x1.4p+6f, -0x1.92a738p-19f, 0x1.55775875ba168p+115, -0x1.14cb4ae10b3a9p+61},
    {fn_log, 0x1p+0f, 0x1.8p-30f, 0x1.7ffffffb8p-30, 0x1.1ffffffafp-90},
    {fn_log, 0x1.fffffep-1f, -0x1.4p-27f, -0x1.280000ab20008p-24, -0x1.f4558e87378a4p-79},
    {fn_log, 0x1.0104d6p+0f, 0x1.79f506p-26f, 0x1.0451d66a075f2p-8, -0x1.af4e3cb93ba94p-62},
    {fn_log, 0x1.051b52p+0f, 0x1.f90daap-25f, 0x1.439d2958f79b6p-6, 0x1.e4d2c9f034603p-61},
    {fn_log, 0x1.535192p+0f, 0x1.2d7fbap-26f, 0x1.2086425903c9dp-2, -0x1.6ff6ee8e08a52p-58},
    {fn_log, 0x1.70d7cap-1f, 0x1.588b04p-27f, -0x1.4fd31a6d990e2p-2, -0x1.2a04df49db92dp-56},
    {fn_log, 0x1.23db1ep+0f, -0x1.50ef84p-25f, 0x1.0c75237f7405cp-3, 0x1.01708579e22d9p-58},
    {fn_log, 0x1.5862b6p+1f, -0x1.229068p-25f, 0x1.fabe1b741cce7p-1, 0x1.75b05aca75f4ep-56},
    {fn_log, 0x1p-1f, 0x0p+0f, -0x1.62e42fefa39efp-1, -0x1.abc9e3b39803fp-56},
    {fn_log, 0x1.4484cp-100f, 0x1.518f64p-125f, -0x1.144f69fd54ee4p+6, -0x1.783221a41a055p-48},
    {fn_log, 0x1.c363ccp+127f, 0x1.15add6p+103f, 0x1.62632b8fb5faep+6, -0x1.08b7646fb1435p-48},
    {fn_log, 0x1p-126f, 0x0p+0f, -0x1.5d589f2fe5107p+6, -0x1.e28d5e1264d1fp-48},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_log, 0x1.16c2p-133f, 0x0p+0f, -0x1.7069e413e07ap+6, 0x1.19e79e767792cp-48},
    {fn_log, 0x1p-149f, 0x0p+0f, -0x1.9d1d9fccf477p+6, -0x1.d0fc8187877a5p-48},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_log2, 0x1p+0f, 0x1.8p-30f, 0x1.14ff58bacb25ep-29, -0x1.1384a794662fep-85},
    {fn_log2, 0x1.fffffep-1f, -0x1.4p-27f, -0x1.ab09a9c6868dbp-24, -0x1.85b5c87cf3302p-79},
    {fn_log2, 0x1.0104d6p+0f, 0x1.79f506p-26f, 0x1.778fd93b05c19p-8, -0x1.9e41a423c3e93p-64},
    {fn_log2, 0x1.051b52p+0f, 0x1.f90daap-25f, 0x1.d2e04dbdd053fp-6, 0x1.c28318f9c90e9p-60},
    {fn_log2, 0x1.535192p+0f, 0x1.2d7fbap-26f, 0x1.a040b70b30d7ep-2, -0x1.562c4bb164ffap-60},
    {fn_log2, 0x1.70d7cap-1f, 0x1.588b04p-27f, -0x1.e47e159476ff2p-2, -0x1.a7c036f01e9ebp-58},
    {fn_log2, 0x1.23db1ep+0f, -0x1.50ef84p-25f, 0x1.834d6aaf1916ep-3, 0x1.13408bc1a0238p-57},
    {fn_log2, 0x1.5862b6p+1f, -0x1.229068p-25f, 0x1.6d899b155ee16p+0, 0x1.65bce75e4c1cfp-56},
    {fn_log2, 0x1p-1f, 0x0p+0f, -0x1p+0, 0x0p+0},
    {fn_log2, 0x1.4484cp-100f, 0x1.518f64p-125f, -0x1.8ea1a18ad1a2dp+6, 0x1.2bee841bef04ap-48},
    {fn_log2, 0x1.c363ccp+127f, 0x1.15add6p+103f, 0x1.ff45de20a9481p+6, -0x1.39546b981509ep-50},
    {fn_log2, 0x1p-126f, 0x0p+0f, -0x1.f8p+6, 0x0p+0},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_log2, 0x1.16c2p-133f, 0x0p+0f, -0x1.09c11763ab996p+7, 0x1.148f9776dd8cbp-47},
    {fn_log2, 0x1p-149f, 0x0p+0f, -0x1.2ap+7, 0x0p+0},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_log10, 0x1p+0f, 0x1.8p-30f, 0x1.4d89c4f9ea1f6p-31, -0x1.22816dd0de19ap-85},
    {fn_log10, 0x1.fffffep-1f, -0x1.4p-27f, -0x1.011a33184aec7p-25, 0x1.7ed8ea63a2746p-79},
    {fn_log10, 0x1.0104d6p+0f, 0x1.79f506p-26f, 0x1.c438bab09237ap-10, -0x1.b8dfde8589262p-65},
    {fn_log10, 0x1.051b52p+0f, 0x1.f90daap-25f, 0x1.1916649b5cb6p-7, 0x1.5c5e143bbef75p-61},
    {fn_log10, 0x1.535192p+0f, 0x1.2d7fbap-26f, 0x1.f537e2e58b696p-4, -0x1.dbf2459403381p-60},
    {fn_log10, 0x1.70d7cap-1f, 0x1.588b04p-27f, -0x1.23b18d2d88b4p-3, 0x1.e9a33703ff006p-65},
    {fn_log10, 0x1.23db1ep+0f, -0x1.50ef84p-25f, 0x1.d25bcb40dc159p-5, 0x1.5d0451aa05c12p-60},
    {fn_log10, 0x1.5862b6p+1f, -0x1.229068p-25f, 0x1.b826aa41cf1a4p-2, 0x1.b7d6c09a05e37p-57},
    {fn_log10, 0x1p-1f, 0x0p+0f, -0x1.34413509f79ffp-2, 0x1.9dc1da994fd21p-59},
    {fn_log10, 0x1.4484cp-100f, 0x1.518f64p-125f, -0x1.dffffffc042e8p+4, 0x1.01109c38c660cp-56},
    {fn_log10, 0x1.c363ccp+127f, 0x1.15add6p+103f, 0x1.33d124f508799p+5, -0x1.aa1183780fc8ep-49},
    {fn_log10, 0x1p-126f, 0x0p+0f, -0x1.2f703035cfc17p+5, 0x1.174ad32eea92cp-52},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_log10, 0x1.16c2p-133f, 0x0p+0f, -0x1.4000013a2d73ep+5, 0x1.edef05b119c5dp-51},
    {fn_log10, 0x1p-149f, 0x0p+0f, -0x1.66d3e7bd9a403p+5, 0x1.c868ea1d9dbaap-50},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_log1p, 0x1p-40f, 0x1.8p-66f, 0x1.0000005fff8p-40, -0x1.7ffeaaf2a92abp-106},
    {fn_log1p, -0x1.8p-51f, 0x1p-77f, -0x1.7fffffc000002p-51, -0x1.fffffa000002cp-106},
    {fn_log1p, 0x1.05557ep-8f, -0x1.f56fc8p-33f, 0x1.04d073d0bbc7p-8, 0x1.c67063d9c2e6bp-62},
    {fn_log1p, -0x1.ff2e48p-9f, 0x1.b7cdfep-34f, -0x1.00170ff786a6fp-8, 0x1.19e04e3a86689p-64},
    {fn_log1p, 0x1.1b955p-4f, 0x1.ef9bdep-29f, 0x1.12327edb1d175p-4, 0x1.7c7524bd6b5f4p-58},
    {fn_log1p, -0x1.0ad1d6p-2f, 0x1.a31c7cp-27f, -0x1.351d803bb2c69p-2, 0x1.69d4b7a1ade8cp-57},
    {fn_log1p, 0x1.333334p-2f, -0x1.2e5d9ep-30f, 0x1.0ca9384d1c09dp-2, -0x1.c505e2684aec4p-57},
    {fn_log1p, 0x1.a3d70ap-2f, 0x0p+0f, 0x1.5fd5fa92d0f7fp-2, 0x1.ec012c0513324p-56},
    {fn_log1p, -0x1p-1f, 0x1.e1094ep-27f, -0x1.62e42eff1ef83p-1, 0x1.0a8b34c83260bp-55},
    {fn_log1p, 0x1.8p+1f, 0x0p+0f, 0x1.62e42fefa39efp+0, 0x1.abc9e3b39803fp-55},
    {fn_log1p, 0x1.2a05f2p+33f, 0x1.9p+6f, 0x1.7069e2ace0b6cp+4, 0x1.1d81a578ea267p-51},
    {fn_log1p, 0x1.57586cp+46f, 0x1.5e295ep+21f, 0x1.016d382fb6b35p+5, -0x1.5831ffc44d395p-49},
    {fn_log1p, -0x1.ffea3ep-1f, 0x1.01f70ap-26f, -0x1.168291f042cc3p+3, 0x1.2b873b5022342p-52},
    {fn_log1p, -0x1.fffffep-1f, -0x1.ffffd2p-26f, -0x1.1542440337e4bp+4, -0x1.86db7e4bc1d79p-50},
    {fn_log1p, -0x1p+0f, 0x1p-30f, -0x1.4cb5ecf0a965p+4, -0x1.0886a2bc2f41ep-50},
    {fn_log1p, 0x1.c363ccp+127f, 0x0p+0f, 0x1.62632b8d400d2p+6, 0x1.bb1ea2cf46fa7p-48},
  };
  for (explog_case c : cases)
  {
    const T r = call(c.fn, T(c.x_hi, c.x_lo));
    assert(rel_err_u2(r, c) <= tolerance[c.fn]);
  }
}

// Below 2^-7 fp32mp2_high takes expm1 and log1p from their Taylor series, which MPFR sweeps put
// within 2.5u^2 there; the other levels err by up to 4.5u^2 (expm1) and 3.9u^2 (log1p). The
// arguments are the worst those sweeps found for the other levels.
TEST_HOST_DEVICE_FUNC void test_series_high()
{
  // MPFR 4.2.2, 256-bit precision
  const explog_case cases[] = {
    {fn_expm1, 0x1.6be27ap-9f, 0x1.fad986p-34f, 0x1.6c63e8a96671bp-9, 0x1.2b6c03b18ad39p-65},
    {fn_expm1, 0x1.666d7cp-9f, 0x1.cbe326p-34f, 0x1.66eb0fcbcb425p-9, 0x1.44bc8cc9003ddp-63},
    {fn_expm1, -0x1.6c3dfap-9f, -0x1.c20d4ap-34f, -0x1.6bbc8980d355dp-9, 0x1.f5d444306830ap-63},
    {fn_expm1, 0x1.70fde8p-9f, 0x1.dd7c2p-34f, 0x1.7182ffa3f9bf2p-9, -0x1.68008db1ac3d8p-63},
    {fn_log1p, 0x1.084e76p-8f, -0x1.b5132cp-33f, 0x1.07c661d1ae85ep-8, 0x1.ed502985e15bfp-62},
    {fn_log1p, 0x1.0dd17ep-8f, -0x1.c623f8p-33f, 0x1.0d43afc13bf6bp-8, -0x1.6f12b93b2eaa5p-65},
    {fn_log1p, -0x1.fd68f4p-8f, -0x1.466588p-33f, -0x1.ff666e34b8a8fp-8, 0x1.a54388685e975p-68},
    {fn_log1p, 0x1.05557ep-8f, -0x1.f56fc8p-33f, 0x1.04d073d0bbc7p-8, 0x1.c67063d9c2e6bp-62},
  };
  for (explog_case c : cases)
  {
    const cuda::fp32mp2_high r = call(c.fn, cuda::fp32mp2_high(c.x_hi, c.x_lo));
    assert(rel_err_u2(r, c) <= 2.5);
  }
}

template <class F>
struct exact_case
{
  explog_fn fn;
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

// Exact results, with a zero low limb: b^0 = 1, exp2 of an integer, log_b(1) = 0, and log2 of a
// power of two.
TEST_HOST_DEVICE_FUNC void test_exact()
{
  const exact_case<float> cases[] = {
    {fn_exp, 0.0f, 0.0f, 1.0f},          {fn_exp, -0.0f, 0.0f, 1.0f},         {fn_exp10, 0.0f, 0.0f, 1.0f},
    {fn_exp2, 0.0f, 0.0f, 1.0f},         {fn_exp2, 1.0f, 0.0f, 2.0f},         {fn_exp2, -1.0f, 0.0f, 0.5f},
    {fn_exp2, 23.0f, 0.0f, 0x1p23f},     {fn_exp2, 127.0f, 0.0f, 0x1p127f},   {fn_exp2, -126.0f, 0.0f, 0x1p-126f},
    {fn_log, 1.0f, 0.0f, 0.0f},          {fn_log10, 1.0f, 0.0f, 0.0f},        {fn_log2, 1.0f, 0.0f, 0.0f},
    {fn_log2, 2.0f, 0.0f, 1.0f},         {fn_log2, 0.5f, 0.0f, -1.0f},        {fn_log2, 1024.0f, 0.0f, 10.0f},
    {fn_log2, 0x1p127f, 0.0f, 127.0f},   {fn_log2, 0x1p-126f, 0.0f, -126.0f},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp2, -140.0f, 0.0f, 0x1p-140f}, {fn_exp2, -149.0f, 0.0f, 0x1p-149f}, {fn_log2, 0x1p-140f, 0.0f, -140.0f},
    {fn_log2, 0x1p-149f, 0.0f, -149.0f},
#endif // !TEST_HAS_CUDA_FAST_MATH
  };
  check_exact<cuda::fp32mp2>(cases, true);
}

template <class T, class F>
TEST_HOST_DEVICE_FUNC void test_special_values()
{
  const F inf  = cuda::std::numeric_limits<F>::infinity();
  const F nan  = cuda::std::numeric_limits<F>::quiet_NaN();
  const F maxn = (cuda::std::numeric_limits<F>::max)();
  const F minn = (cuda::std::numeric_limits<F>::min)();
#if TEST_HAS_CUDA_FAST_MATH
  // --use_fast_math flushes subnormals to zero on the device (FTZ): the smallest normal stands
  // in for the smallest subnormal there
  const F mind = minn;
#else
  const F mind = cuda::std::numeric_limits<F>::denorm_min();
#endif // TEST_HAS_CUDA_FAST_MATH
  const exact_case<F> cases[] = {
    // exp, exp2, exp10: NaN, +inf, and +0 at -inf; overflow and underflow far beyond the range;
    // exp of a tiny x is 1
    {fn_exp, nan, F(0), nan},
    {fn_exp, -nan, F(0), nan},
    {fn_exp, inf, F(0), inf},
    {fn_exp, -inf, F(0), F(0)},
    {fn_exp, maxn, F(0), inf},
    {fn_exp, -maxn, F(0), F(0)},
    {fn_exp, mind, F(0), F(1)},
    {fn_exp, -minn, F(0), F(1)},
    {fn_exp2, nan, F(0), nan},
    {fn_exp2, inf, F(0), inf},
    {fn_exp2, -inf, F(0), F(0)},
    {fn_exp2, maxn, F(0), inf},
    {fn_exp2, -maxn, F(0), F(0)},
    {fn_exp10, nan, F(0), nan},
    {fn_exp10, inf, F(0), inf},
    {fn_exp10, -inf, F(0), F(0)},
    {fn_exp10, maxn, F(0), inf},
    {fn_exp10, -maxn, F(0), F(0)},
    // expm1: x itself for a zero or tiny x, -1 at -inf
    {fn_expm1, nan, F(0), nan},
    {fn_expm1, inf, F(0), inf},
    {fn_expm1, -inf, F(0), F(-1)},
    {fn_expm1, maxn, F(0), inf},
    {fn_expm1, -maxn, F(0), F(-1)},
    {fn_expm1, F(0), F(0), F(0)},
    {fn_expm1, -F(0), F(0), -F(0)},
    {fn_expm1, mind, F(0), mind},
    {fn_expm1, -mind, F(0), -mind},
    {fn_expm1, -minn, F(0), -minn},
    // log, log2, log10: -inf at +-0, NaN below zero, +inf at +inf
    {fn_log, nan, F(0), nan},
    {fn_log, -nan, F(0), nan},
    {fn_log, inf, F(0), inf},
    {fn_log, -inf, F(0), nan},
    {fn_log, F(0), F(0), -inf},
    {fn_log, -F(0), F(0), -inf},
    {fn_log, -mind, F(0), nan},
    {fn_log, -minn, F(0), nan},
    {fn_log, F(-1), F(0), nan},
    {fn_log, -maxn, F(0), nan},
    {fn_log2, nan, F(0), nan},
    {fn_log2, inf, F(0), inf},
    {fn_log2, -inf, F(0), nan},
    {fn_log2, F(0), F(0), -inf},
    {fn_log2, -F(0), F(0), -inf},
    {fn_log2, -mind, F(0), nan},
    {fn_log2, F(-2), F(0), nan},
    {fn_log10, nan, F(0), nan},
    {fn_log10, inf, F(0), inf},
    {fn_log10, -inf, F(0), nan},
    {fn_log10, F(0), F(0), -inf},
    {fn_log10, -F(0), F(0), -inf},
    {fn_log10, -mind, F(0), nan},
    {fn_log10, F(-10), F(0), nan},
    // log1p: x itself for a zero or tiny x, -inf at -1, NaN below -1
    {fn_log1p, nan, F(0), nan},
    {fn_log1p, inf, F(0), inf},
    {fn_log1p, -inf, F(0), nan},
    {fn_log1p, F(-1), F(0), -inf},
    {fn_log1p, F(-1.5), F(0), nan},
    {fn_log1p, -maxn, F(0), nan},
    {fn_log1p, F(0), F(0), F(0)},
    {fn_log1p, -F(0), F(0), -F(0)},
    {fn_log1p, mind, F(0), mind},
    {fn_log1p, -mind, F(0), -mind},
    {fn_log1p, -minn, F(0), -minn},
  };
  check_exact<T>(cases, false);
}

// Results at and beyond the ends of the fp32mp2 range: overflow to +inf, including the float
// nearest ln(2^128), whose exponential is above 2^128; subnormal results rounded once; results
// below half the smallest subnormal; and log1p of an x that only its low limb puts below -1.
TEST_HOST_DEVICE_FUNC void test_range_ends()
{
  const float inf                 = cuda::std::numeric_limits<float>::infinity();
  const float nan                 = cuda::std::numeric_limits<float>::quiet_NaN();
  const exact_case<float> cases[] = {
    {fn_exp, 0x1.62e43p+6f, 0.0f, inf},
    {fn_exp, 0x1.62e43p+6f, 0x1p-18f, inf},
    {fn_exp, 89.0f, 0.0f, inf},
    {fn_exp, 1000.0f, 0.0f, inf},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp, -100.0f, 0.0f, 0x1.bp-145f}, // 26.5 subnormal ulps, rounded to 27
    {fn_exp, -103.5f, 0.0f, 0x1p-149f},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_exp, -104.0f, 0.0f, 0.0f},
    {fn_exp, -1000.0f, 0.0f, 0.0f},
    {fn_exp2, 128.0f, 0.0f, inf},
    {fn_exp2, 128.0f, -0x1p-24f, 0x1.fffffep+127f}, // 2^128 (1 - 4e-8) rounds to FLT_MAX
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp2, -126.5f, 0.0f, 0x1.6a09e8p-127f},
    {fn_exp2, -149.5f, 0.0f, 0x1p-149f}, // 2^-149.5 rounds up to the smallest subnormal
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_exp2, -150.5f, 0.0f, 0.0f}, // and 2^-150.5 down to zero
    {fn_exp2, -1000.0f, 0.0f, 0.0f},
    {fn_exp10, 39.0f, 0.0f, inf},
    {fn_exp10, 1000.0f, 0.0f, inf},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {fn_exp10, -45.0f, 0.0f, 0x1p-149f}, // 0.71 subnormal ulps
#endif // !TEST_HAS_CUDA_FAST_MATH
    {fn_exp10, -46.0f, 0.0f, 0.0f},
    {fn_expm1, 0x1.62e43p+6f, 0.0f, inf},
    {fn_expm1, 1000.0f, 0.0f, inf},
    {fn_expm1, -41.0f, 0.0f, -1.0f},
    {fn_expm1, -1000.0f, 0.0f, -1.0f},
    {fn_log1p, -1.0f, -0x1p-30f, nan},
  };
  check_exact<cuda::fp32mp2>(cases, false);
}

int main(int, char**)
{
  test_accuracy<cuda::fp32mp2>();
  test_accuracy<cuda::fp32mp2_low>();
  test_accuracy<cuda::fp32mp2_high>();
  test_series_high();
  test_exact();
  test_special_values<cuda::fp32mp2, float>();
  test_special_values<cuda::fp32mp2_high, float>();
  test_special_values<cuda::fp64mp2, double>();
  test_special_values<cuda::fp64mp2_high, double>();
  test_range_ends();

  return 0;
}
