// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

// fp32mp2 asin, acos, atan and atan2: within a few u^2 (u = 2^-24) of an MPFR reference over
// the whole range of each, including small arguments, x near +-1 for asin and acos (where acos
// is small), every octant of atan2, and atan2 operands near both ends of the exponent range,
// subnormal ones included (which gave NaN or lost the low limb); and the C99 Annex F special
// values, with the sign of zero. Every accuracy level is checked, and the special values for
// fp64mp2 as well, and so are the fp64mp2 asin and acos next to the ends of the domain.

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

enum invtrig_fn
{
  fn_asin,
  fn_acos,
  fn_atan,
  fn_count
};

template <class T>
TEST_HOST_DEVICE_FUNC T call(invtrig_fn fn, T x)
{
  switch (fn)
  {
    case fn_asin:
      return cuda::asin(x);
    case fn_acos:
      return cuda::acos(x);
    default:
      return cuda::atan(x);
  }
}

struct unary_case
{
  invtrig_fn fn;
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

struct atan2_case
{
  float y_hi;
  float y_lo;
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
template <class T>
TEST_HOST_DEVICE_FUNC double rel_err_u2(T r, double ref_hi, double ref_lo)
{
  double d = ((static_cast<double>(r.hi()) - ref_hi) + static_cast<double>(r.lo())) - ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(ref_hi) * 0x1p48;
}

// `tolerance` holds the bounds for asin, acos, atan and atan2, in that order.
template <class T>
TEST_HOST_DEVICE_FUNC void test_accuracy(const double* tolerance)
{
  const double atan2_tolerance = tolerance[fn_count];
  const float inf              = cuda::std::numeric_limits<float>::infinity();
  // MPFR 4.2.2, 256-bit precision: the function, x, and its value. Small arguments, the worst
  // arguments of the sweeps, arguments around 1/32, and the ends of each range, where asin and
  // atan reach +-pi/2.
  const unary_case cases[] = {
    {fn_asin, 0x1p-40f, 0x1.8p-66f, 0x1.0000006p-40, 0x1.555556d55555ep-123},
    {fn_asin, -0x1.8p-100f, 0x1p-126f, -0x1.7fffffcp-100, -0x1.1fffff7000002p-301},
    {fn_asin, 0x1.7f62b6p-5f, 0x1.2e5d9ep-30f, 0x1.7f869379dd7p-5, 0x1.79156bcac9b02p-59},
    {fn_asin, 0x1.83fc26p-5f, 0x1.f9217cp-30f, 0x1.8421528012b2bp-5, -0x1.8cb90a60a834bp-60},
    {fn_asin, -0x1.6debbap-3f, 0x1.848f8p-33f, -0x1.6fe57082ad52bp-3, -0x1.80730f666b583p-57},
    {fn_asin, 0x1.280354p-2f, 0x1.fc96a4p-27f, 0x1.2c4c8601338aap-2, -0x1.7fa7753e2a7adp-56},
    {fn_asin, 0x1p-1f, 0x0p+0f, 0x1.0c152382d7366p-1, -0x1.ee6913347c2a6p-55},
    {fn_asin, -0x1.11826ap-1f, 0x1.3ab114p-28f, -0x1.208ac9df3f27p-1, -0x1.cc08a2b5a95c8p-55},
    {fn_asin, -0x1.2bafecp-1f, -0x1.e1e04cp-27f, -0x1.4025103895667p-1, 0x1.b3215af1433ecp-55},
    {fn_asin, -0x1.457c6ap-1f, -0x1.221724p-26f, -0x1.60bbfc97ba4f6p-1, -0x1.096e22ca3cedbp-56},
    {fn_asin, 0x1.6a09e6p-1f, 0x1.9fcef4p-27f, 0x1.921fb54442d1dp-1, 0x1.12ae4ff8edc0ep-56},
    {fn_asin, -0x1.c8b07ap-1f, 0x1.23aaa2p-26f, -0x1.1a082be2fcb8dp+0, -0x1.4e9b5dd41fcd7p-54},
    {fn_asin, 0x1.e9be92p-1f, -0x1.57a058p-26f, 0x1.465dcc09feba9p+0, 0x1.6a48b7d2433e9p-55},
    {fn_asin, 0x1.fffffep-1f, 0x1.4p-27f, 0x1.920aec6813193p+0, -0x1.247539d9f408ap-54},
    {fn_asin, 0x1p+0f, -0x1p-40f, 0x1.921f9ea3a46bp+0, 0x1.4af6911701d82p-54},
    {fn_asin, -0x1p+0f, 0x1p-100f, -0x1.921fb54442d13p+0, 0x1.863c034df60c2p-54},
    {fn_asin, 0x1p+0f, 0x0p+0f, 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54},
    {fn_asin, -0x1p+0f, 0x0p+0f, -0x1.921fb54442d18p+0, -0x1.1a62633145c07p-54},
    {fn_acos, 0x1p-40f, 0x1.8p-66f, 0x1.921fb54441d18p+0, 0x1.1a4a633145c07p-54},
    {fn_acos, 0x0p+0f, 0x0p+0f, 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54},
    {fn_acos, -0x1.8p-100f, 0x1p-126f, 0x1.921fb54442d18p+0, 0x1.1a62633145c67p-54},
    {fn_acos, 0x1.c3602ap-2f, 0x1.6fd77p-29f, 0x1.1d43733b28b5bp+0, -0x1.a0b02506d76fap-55},
    {fn_acos, 0x1.fe538ap-2f, -0x1.b54884p-32f, 0x1.0c90c1ed203cdp+0, -0x1.dc32b6963ada7p-56},
    {fn_acos, -0x1.20038ep-1f, 0x1.876f12p-27f, 0x1.1588bdecc3eabp+1, 0x1.366bf28e80cc3p-53},
    {fn_acos, 0x1.6a09e6p-1f, 0x1.9fcef4p-27f, 0x1.921fb54442d13p-1, 0x1.ab6d9e6614a07p-55},
    {fn_acos, 0x1.6fa6f4p-1f, 0x1.6827a4p-26f, 0x1.8a1f50a420096p-1, -0x1.f5e2a14dbabe1p-59},
    {fn_acos, 0x1.b0b586p-1f, -0x1.9007f2p-26f, 0x1.20c1d68cb564ep-1, 0x1.f91e2f30cd55bp-57},
    {fn_acos, 0x1.dddab6p-1f, -0x1.6d08e8p-26f, 0x1.7816f933cfdd1p-2, -0x1.ad1b38f8b5691p-58},
    {fn_acos, 0x1.eee7e6p-1f, -0x1.456854p-29f, 0x1.0959a94cc57fdp-2, 0x1.45141f8fcd71cp-57},
    {fn_acos, 0x1.fed56cp-1f, 0x1.764012p-26f, 0x1.148594b19dd6cp-4, 0x1.a91044a284207p-60},
    {fn_acos, 0x1.fffffep-1f, 0x1.4p-27f, 0x1.4c8dc2fb858fbp-12, 0x1.79d0b39c90a9cp-66},
    {fn_acos, 0x1p+0f, -0x1p-40f, 0x1.6a09e667f3dafp-20, 0x1.21a43e8525b27p-74},
    {fn_acos, 0x1p+0f, -0x1p-100f, 0x1.6a09e667f3bcdp-50, -0x1.bdd3413b2644ep-104},
    {fn_acos, 0x1p+0f, -0x1p-140f, 0x1.6a09e667f3bcdp-70, -0x1.bdd3413b26456p-124},
    {fn_acos, -0x1p+0f, 0x1p-40f, 0x1.921fa9f3f39e4p+1, 0x1.32ac7a2423cc4p-53},
    {fn_acos, -0x1p-1f, 0x0p+0f, 0x1.0c152382d7366p+1, -0x1.ee6913347c2a6p-53},
    {fn_acos, -0x1p+0f, 0x0p+0f, 0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53},
    {fn_atan, 0x1p-40f, 0x1.8p-66f, 0x1.0000006p-40, -0x1.555556d55555ep-122},
    {fn_atan, -0x1.8p-100f, 0x1p-126f, -0x1.7fffffcp-100, 0x1.1fffff7000002p-300},
    {fn_atan, 0x1.7f62b6p-5f, 0x1.2e5d9ep-30f, 0x1.7f1b26fdcf748p-5, 0x1.4c62b1584cd1ep-60},
    {fn_atan, -0x1.091e06p-4f, -0x1.3cd42p-29f, -0x1.08bf7bf4c83c7p-4, -0x1.57a6e32ff4daep-58},
    {fn_atan, -0x1.43a57ap-2f, -0x1.c18958p-28f, -0x1.397903e9d4a35p-2, 0x1.b697b60281eefp-57},
    {fn_atan, -0x1.e8b0eap-2f, 0x1.f40c1cp-27f, -0x1.c7f571ffd7951p-2, 0x1.2b092e699065ep-58},
    {fn_atan, -0x1.7c934ep-1f, 0x1.e9771ep-26f, -0x1.474616109ca51p-1, -0x1.2c4fb8ab29a5dp-57},
    {fn_atan, -0x1.f9860ap-1f, -0x1.6c4b04p-26f, -0x1.8edd76c37a3c4p-1, 0x1.5c0e7da207fefp-55},
    {fn_atan, 0x1p+0f, 0x0p+0f, 0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55},
    {fn_atan, -0x1.013ef6p+0f, 0x1.fa1f62p-25f, -0x1.935de3e6e981ep-1, -0x1.1dac7dde5db88p-55},
    {fn_atan, -0x1.c20502p+0f, -0x1.cef0bp-25f, -0x1.0db7c80f2b07dp+0, -0x1.97e8c3704432fp-56},
    {fn_atan, 0x1.027bb4p+1f, 0x1.d5bb6ep-24f, 0x1.1c6a6bf768308p+0, -0x1.f30e75bc0b44dp-58},
    {fn_atan, -0x1.19471cp+4f, -0x1.4de56p-21f, -0x1.8393d1aff5e39p+0, 0x1.06ae753aeec5dp-57},
    {fn_atan, 0x1.08p+5f, 0x1.0c6f7ap-20f, 0x1.8a5e605413ccp+0, 0x1.a91b8fb219a89p-56},
    {fn_atan, 0x1.ec0e6ep+30f, -0x1.3c7414p+5f, 0x1.921fb5422e11p+0, 0x1.34e9430fdfb7ap-55},
    {fn_atan, 0x1p+59f, 0x1p+30f, 0x1.921fb54442d18p+0, 0x1.1262633185c07p-54},
    {fn_atan, -0x1p+61f, 0x0p+0f, -0x1.921fb54442d18p+0, -0x1.1862633145c07p-54},
    {fn_atan, 0x1.c363ccp+127f, 0x1.15add6p+103f, 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54},
    {fn_atan, inf, 0x0p+0f, 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54},
    {fn_atan, -inf, 0x0p+0f, -0x1.921fb54442d18p+0, -0x1.1a62633145c07p-54},
    // the worst arguments of the sweeps of the polynomial kernels
    {fn_asin, 0x1.332cdcp-1f, 0x1.0e19d4p-30f, 0x1.49710d48bddcfp-1, -0x1.71ad4c437d8c2p-60},
    {fn_asin, 0x1.0426dp-1f, 0x1.f791e4p-26f, 0x1.10e3ba02050bp-1, -0x1.e83fc97ba1343p-55},
    {fn_acos, 0x1.b631ccp-1f, 0x1.a2c938p-29f, 0x1.1653a47f86146p-1, 0x1.b66ad1b8c8c9ap-60},
    {fn_acos, 0x1.b8352p-1f, 0x1.dba91ep-27f, 0x1.1268f431612c7p-1, -0x1.92348da984f16p-55},
    {fn_acos, 0x1.edcca8p-1f, -0x1.4632d2p-29f, 0x1.11da58dddcb09p-2, 0x1.a28387dfe501p-60},
    {fn_atan, 0x1.2bf338p-1f, 0x1.30ad34p-28f, 0x1.0f54a4f804d21p-1, 0x1.b5f28771df89bp-55},
    {fn_atan, 0x1.de254ep+0f, 0x1.fbd1a8p-25f, 0x1.14486cfd0cda2p+0, 0x1.a5abe4c0ffa9dp-54},
  };
  for (unary_case c : cases)
  {
    const T r = call(c.fn, T(c.x_hi, c.x_lo));
    assert(rel_err_u2(r, c.ref_hi, c.ref_lo) <= tolerance[c.fn]);
  }
  // y, x, and atan2(y, x): the worst arguments of the sweeps, every octant, both sides of the
  // diagonal, ratios at both ends of the range, and operands near the ends of the exponent
  // range, subnormal ones included.
  const atan2_case cases2[] = {
    {0x1.03a3c8p-1f, -0x1.f8034ep-26f, 0x1.7fc9fp+0f, 0x1.eb29fp-25f, 0x1.4e014632613fap-2, 0x1.78b0574a62c8p-56},
    {-0x1.04629ap-1f, 0x1.a764d2p-26f, 0x1.0298dcp-1f, 0x1.a5e24ep-26f, -0x1.93e348a3fda19p-1, -0x1.d32b802a42778p-56},
    {0x1.114e6ap-1f, 0x1.ee5842p-26f, 0x1.0a19bp+2f, -0x1.0d4e98p-23f, 0x1.0580818f95adep-3, -0x1.0070c417941c8p-58},
    {0x1.8ab8d2p-4f, 0x1.5ab932p-29f, 0x1.29574ap+2f, -0x1.fa65b8p-23f, 0x1.53caf8f737016p-6, -0x1.942b0da2440e4p-62},
    {-0x1.160fd6p-3f, -0x1.5bd3a2p-28f, 0x1.676cap+2f, -0x1.c1dcc2p-23f, -0x1.8c05a5320b045p-6, 0x1.aa12190e93399p-60},
    {0x1.176d3ep-17f, 0x1.0adfbcp-42f, 0x1.034736p-12f, -0x1.e59becp-37f, 0x1.13ca23063f1a5p-5, -0x1.86e0851875fa5p-62},
    {0x1.8p+1f, 0x1.ad7f2ap-24f, -0x1p+2f, 0x1.421f6p-23f, 0x1.3fc1766ea4a34p+1, -0x1.cb6467d83e6dp-53},
    {-0x1.8p+1f, 0x1.ad7f2ap-24f, -0x1p+2f, 0x1.421f6p-23f, -0x1.3fc176b35cd2fp+1, -0x1.80fb6b5079c7bp-54},
    {0x1p+2f, -0x1.421f6p-23f, -0x1.8p+1f, 0x1.ad7f2ap-24f, 0x1.1b6e193307675p+1, 0x1.ce2bbe435779ap-54},
    {-0x1p+2f, -0x1.421f6p-23f, 0x1.8p+1f, 0x1.ad7f2ap-24f, -0x1.dac6706749c11p-1, -0x1.48135b0dd8b75p-55},
    {0x1p+0f, 0x0p+0f, 0x1p+0f, 0x0p+0f, 0x1.921fb54442d18p-1, 0x1.1a62633145c07p-55},
    {0x1p+0f, 0x1p-30f, -0x1p+0f, 0x1p-31f, 0x1.2d97c7f1b21d2p+1, 0x1.a85394c9ea20ap-54},
    {0x1p-40f, 0x1.8p-66f, 0x1p+0f, 0x0p+0f, 0x1.0000006p-40, -0x1.555556d55555ep-122},
    {0x1p-40f, 0x1.8p-66f, -0x1p+0f, 0x0p+0f, 0x1.921fb54442518p+1, 0x1.1a56633145c07p-53},
    {0x1p+0f, 0x0p+0f, 0x1p-40f, 0x1.8p-66f, 0x1.921fb54441d18p+0, 0x1.1a4a633145c07p-54},
    {-0x1p+0f, 0x0p+0f, -0x1p-40f, 0x1.8p-66f, -0x1.921fb54443d18p+0, -0x1.1a4a633145c07p-54},
    {0x1.1b3a64p-99f, 0x1.5e6bap-124f, 0x1.148f82p-90f, -0x1.e2b75p-115f, 0x1.062be72d6077bp-9, 0x1.918b128fdcc2bp-66},
    {0x1.1190ep-126f, 0x0p+0f, 0x1.0c2acep-117f, -0x1.f8p-142f, 0x1.052747583804fp-9, -0x1.9ad05263cae75p-63},
    {-0x1.2b4be6p-126f, 0x0p+0f, 0x1.07edcp-33f, -0x1.abb3fp-58f, -0x1.224e27c6a3058p-93, -0x1.a8d2c0a985f69p-147},
    {0x1p-149f, 0x0p+0f, 0x1p-148f, 0x0p+0f, 0x1.dac670561bb4fp-2, 0x1.a2b7f222f65e2p-56},
    {0x1.4p-140f, 0x0p+0f, -0x1.cp-141f, 0x0p+0f, 0x1.173c1f5810bb8p+1, -0x1.fa63753b6372ep-53},
    {0x1.2p-130f, 0x1p-149f, 0x1p-100f, 0x1.8p-126f, 0x1.20001f93fff43p-30, -0x1.e1e4e46bddbbep-84},
    {0x1.682978p+61f,
     0x1.f3d62cp+36f,
     0x1.41e276p+115f,
     -0x1.86f678p+90f,
     0x1.1e71505c17165p-54,
     -0x1.b97f3e8fe1136p-109},
    {0x1.8p+127f, 0x1p+100f, 0x1.fffffep+127f, 0x0p+0f, 0x1.4978fb3ca75f6p-1, -0x1.de2e147a73c5ap-55},
    {0x1.fffffep+127f, 0x0p+0f, -0x1.8p+127f, 0x1p+100f, 0x1.1b6e19670dcffp+1, 0x1.a18ba5a1990f9p-55},
    {0x1p+100f, 0x1p+70f, 0x1p-100f, 0x0p+0f, 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54},
    {0x1p-100f, 0x0p+0f, -0x1p+100f, 0x1p+70f, 0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53},
    {0x1.8p-20f, 0x1p-46f, 0x1.4p+70f, 0x1p+44f, 0x1.33333328f5c29p-90, 0x1.6872afb7e9101p-146},
    // the worst arguments of the sweeps of the polynomial kernel
    {-0x1.373bf2p+2f, 0x1.674224p-23f, 0x1.0029a2p+3f, 0x1.f968fp-22f, -0x1.1781a6850f628p-1, -0x1.862dc6c1f97bcp-55},
    {0x1.004c32p-28f, -0x1.d8b1e8p-53f, 0x1.eda682p-3f, 0x1.e1d7aep-28f, 0x1.09d312dc06c99p-26, -0x1.251483053c965p-80},
    {0x1.00baecp-107f, -0x1.9a02p-132f, 0x1.ffa1bep-66f, 0x1.aea9bp-91f, 0x1.00ea36e5373bdp-42, 0x1.847fbc674ad78p-97},
  };
  for (atan2_case c : cases2)
  {
    const T r = cuda::atan2(T(c.y_hi, c.y_lo), T(c.x_hi, c.x_lo));
    assert(rel_err_u2(r, c.ref_hi, c.ref_lo) <= atan2_tolerance);
  }
}

template <class F>
TEST_HOST_DEVICE_FUNC bool same(F x, F y)
{
  if (cuda::std::isnan(y))
  {
    return cuda::std::isnan(x);
  }
  return x == y && cuda::std::signbit(x) == cuda::std::signbit(y);
}

// r is k pi/4 for a nonzero k: the high limb to within 4 ulp (fp64mp2 has a binary64 fallback
// whose library is not correctly rounded), and the sign.
template <class T, class F>
TEST_HOST_DEVICE_FUNC bool is_quarter_turns(T r, int k)
{
  const double ref = 0.78539816339744830962 * k;
  const double eps = static_cast<double>(cuda::std::numeric_limits<F>::epsilon());
  return cuda::std::fabs(static_cast<double>(r.hi()) - ref) <= 4.0 * eps * cuda::std::fabs(ref)
      && cuda::std::signbit(r.hi()) == (k < 0);
}

template <class F>
struct exact_case
{
  invtrig_fn fn;
  F x_hi;
  F x_lo;
  F r; // expected high limb, compared bit for bit (NaN matches any NaN)
};

// asin and atan of a zero or tiny x is x itself, sign included; NaN, and for asin and acos
// any |x| > 1, gives NaN in both limbs, also when only the low limb puts |x| above 1.
template <class T, class F>
TEST_HOST_DEVICE_FUNC void test_special_values()
{
  const F inf                 = cuda::std::numeric_limits<F>::infinity();
  const F nan                 = cuda::std::numeric_limits<F>::quiet_NaN();
  const F maxn                = cuda::std::numeric_limits<F>::max();
  const F minn                = cuda::std::numeric_limits<F>::min();
  const F mind                = cuda::std::numeric_limits<F>::denorm_min();
  const F above1              = F(1) + cuda::std::numeric_limits<F>::epsilon();
  const exact_case<F> cases[] = {
    {fn_asin, nan, F(0), nan},     {fn_asin, -nan, F(0), nan},    {fn_asin, F(0), F(0), F(0)},
    {fn_asin, -F(0), F(0), -F(0)}, {fn_asin, mind, F(0), mind},   {fn_asin, -mind, F(0), -mind},
    {fn_asin, minn, F(0), minn},   {fn_asin, -minn, F(0), -minn}, {fn_asin, above1, F(0), nan},
    {fn_asin, -above1, F(0), nan}, {fn_asin, F(1), mind, nan},    {fn_asin, F(-1), -mind, nan},
    {fn_asin, F(2), F(0), nan},    {fn_asin, maxn, F(0), nan},    {fn_asin, -maxn, F(0), nan},
    {fn_asin, inf, F(0), nan},     {fn_asin, -inf, F(0), nan},    {fn_acos, nan, F(0), nan},
    {fn_acos, -nan, F(0), nan},    {fn_acos, F(1), F(0), F(0)},   {fn_acos, above1, F(0), nan},
    {fn_acos, -above1, F(0), nan}, {fn_acos, F(1), mind, nan},    {fn_acos, F(-1), -mind, nan},
    {fn_acos, F(-2), F(0), nan},   {fn_acos, maxn, F(0), nan},    {fn_acos, -maxn, F(0), nan},
    {fn_acos, inf, F(0), nan},     {fn_acos, -inf, F(0), nan},    {fn_atan, nan, F(0), nan},
    {fn_atan, -nan, F(0), nan},    {fn_atan, F(0), F(0), F(0)},   {fn_atan, -F(0), F(0), -F(0)},
    {fn_atan, mind, F(0), mind},   {fn_atan, -mind, F(0), -mind}, {fn_atan, minn, F(0), minn},
    {fn_atan, -minn, F(0), -minn},
  };
  for (const exact_case<F>& c : cases)
  {
    const T r = call(c.fn, T(c.x_hi, c.x_lo));
    assert(same(r.hi(), c.r));
    if (cuda::std::isnan(c.r))
    {
      assert(cuda::std::isnan(r.lo()));
    }
    else
    {
      assert(r.lo() == F(0));
    }
  }

  // +-pi/2 at the ends of the domain, and pi at acos(-1)
  assert((is_quarter_turns<T, F>(cuda::asin(T(F(1))), 2)));
  assert((is_quarter_turns<T, F>(cuda::asin(T(F(-1))), -2)));
  assert((is_quarter_turns<T, F>(cuda::acos(T(F(0))), 2)));
  assert((is_quarter_turns<T, F>(cuda::acos(T(-F(0))), 2)));
  assert((is_quarter_turns<T, F>(cuda::acos(T(F(-1))), 4)));
  assert((is_quarter_turns<T, F>(cuda::atan(T(inf)), 2)));
  assert((is_quarter_turns<T, F>(cuda::atan(T(-inf)), -2)));
  assert((is_quarter_turns<T, F>(cuda::atan(T(maxn)), 2)));
  assert((is_quarter_turns<T, F>(cuda::atan(T(-maxn)), -2)));
}

template <class F>
struct atan2_special
{
  F y;
  F x;
  int k; // expected result: k pi/4, with the sign of y for k = 0
};

// C99 F.10.1.4, with zeros, subnormals, the largest finite value and infinities on either
// side. The sign of the result is the sign of y, also for a zero y.
template <class T, class F>
TEST_HOST_DEVICE_FUNC void test_atan2_special_values()
{
  const F inf                    = cuda::std::numeric_limits<F>::infinity();
  const F nan                    = cuda::std::numeric_limits<F>::quiet_NaN();
  const F maxn                   = cuda::std::numeric_limits<F>::max();
  const F minn                   = cuda::std::numeric_limits<F>::min();
  const F mind                   = cuda::std::numeric_limits<F>::denorm_min();
  const F zero                   = F(0);
  const atan2_special<F> cases[] = {
    // y = +-0: +-0 for x > 0 or +0, +-pi for x < 0 or -0
    {zero, zero, 0},
    {zero, mind, 0},
    {zero, minn, 0},
    {zero, F(1), 0},
    {zero, maxn, 0},
    {zero, inf, 0},
    {zero, -zero, 4},
    {zero, -mind, 4},
    {zero, -minn, 4},
    {zero, F(-1), 4},
    {zero, -maxn, 4},
    {zero, -inf, 4},
    // x = +-0 and y != 0: +-pi/2
    {mind, zero, 2},
    {mind, -zero, 2},
    {minn, zero, 2},
    {F(1), -zero, 2},
    {maxn, zero, 2},
    // y infinite: +-pi/2 for finite x, +-pi/4 for +inf, +-3pi/4 for -inf
    {inf, zero, 2},
    {inf, -zero, 2},
    {inf, mind, 2},
    {inf, F(-1), 2},
    {inf, maxn, 2},
    {inf, -maxn, 2},
    {inf, inf, 1},
    {inf, -inf, 3},
    // x infinite and y finite: +-0 for +inf, +-pi for -inf
    {mind, inf, 0},
    {F(1), inf, 0},
    {maxn, inf, 0},
    {mind, -inf, 4},
    {F(1), -inf, 4},
    {maxn, -inf, 4},
    // subnormal and extreme finite operands
    {mind, mind, 1},
    {mind, -mind, 3},
    {minn, minn, 1},
    {maxn, maxn, 1},
    {maxn, -maxn, 3},
    {mind, maxn, 0},
    {mind, -maxn, 4},
    {maxn, mind, 2},
    {maxn, -mind, 2},
    {mind, F(-1), 4},
    {F(1), mind, 2},
  };
  for (const atan2_special<F>& c : cases)
  {
    for (int negate = 0; negate < 2; ++negate)
    {
      const F y = negate ? -c.y : c.y;
      const T r = cuda::atan2(T(y), T(c.x));
      if (c.k == 0)
      {
        assert(same(r.hi(), negate ? -zero : zero));
        assert(r.lo() == zero);
      }
      else
      {
        assert((is_quarter_turns<T, F>(r, negate ? -c.k : c.k)));
      }
    }
  }

  const F operands[] = {zero, -zero, mind, F(-1), maxn, inf, -inf, nan};
  for (F v : operands)
  {
    const T a = cuda::atan2(T(nan), T(v));
    const T b = cuda::atan2(T(v), T(nan));
    assert(cuda::std::isnan(a.hi()) && cuda::std::isnan(a.lo()));
    assert(cuda::std::isnan(b.hi()) && cuda::std::isnan(b.lo()));
  }
}

// fp64mp2 asin and acos of |x| = 1 - t for a t that only the low limb holds, down to the smallest
// subnormal: acos(1 - t) = sqrt(2 t) to first order, asin(1 - t) = pi/2 - acos(1 - t) and
// acos(-(1 - t)) = pi - acos(1 - t). The high limb is checked to a few ulp, which holds for the
// binary128 and for the binary64 bodies; summing the limbs first gave 0, pi/2 and pi.
TEST_HOST_DEVICE_FUNC void test_fp64mp2_near_one()
{
  const double pi  = 3.14159265358979323846;
  const double t[] = {0x1p-60, 0x1p-70, 0x1p-200, 0x1p-1000, cuda::std::numeric_limits<double>::denorm_min()};
  for (double ti : t)
  {
    const double s = cuda::std::sqrt(2.0) * cuda::std::sqrt(ti);

    const cuda::fp64mp2 c = cuda::acos(cuda::fp64mp2(1.0, -ti));
    assert(cuda::std::fabs(c.hi() - s) <= 1e-13 * s);

    const cuda::fp64mp2 cn = cuda::acos(cuda::fp64mp2(-1.0, ti));
    assert(cuda::std::fabs(cn.hi() - (pi - s)) <= 2e-15);

    const cuda::fp64mp2 a = cuda::asin(cuda::fp64mp2(1.0, -ti));
    assert(cuda::std::fabs(a.hi() - (0.5 * pi - s)) <= 1e-15);

    const cuda::fp64mp2 an = cuda::asin(cuda::fp64mp2(-1.0, ti));
    assert(cuda::std::fabs(an.hi() + (0.5 * pi - s)) <= 1e-15);
  }
}

int main(int, char**)
{
  // asin, acos, atan, atan2: the largest error MPFR sweeps of 2M to 4M arguments per range
  // found, on the host and on an sm_89 device, rounded up: 2.8 for asin, 3.4 for acos, 1.9
  // for atan, 3.4 for atan2.
  const double tolerance[fn_count + 1] = {4.0, 4.5, 3.0, 4.5};
  test_accuracy<cuda::fp32mp2>(tolerance);
  test_accuracy<cuda::fp32mp2_low>(tolerance);
  test_accuracy<cuda::fp32mp2_high>(tolerance);
  test_fp64mp2_near_one();
  test_special_values<cuda::fp32mp2, float>();
  test_special_values<cuda::fp32mp2_low, float>();
  test_special_values<cuda::fp32mp2_high, float>();
  test_special_values<cuda::fp64mp2, double>();
  test_special_values<cuda::fp64mp2_high, double>();
  test_atan2_special_values<cuda::fp32mp2, float>();
  test_atan2_special_values<cuda::fp32mp2_low, float>();
  test_atan2_special_values<cuda::fp32mp2_high, float>();
  test_atan2_special_values<cuda::fp64mp2, double>();
  test_atan2_special_values<cuda::fp64mp2_high, double>();

  return 0;
}
