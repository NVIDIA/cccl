// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 sin, cos, tan and sincos across the argument reduction
//  (no reduction below pi/4, Cody-Waite up to 2^20 including its largest
//  reduced argument, Payne-Hanek above), sinpi, cospi and sincospi across
//  their integer / fraction split, and the IEEE special values, at every
//  accuracy level: fpmp2_accuracy::low and the default evaluate the minimax
//  kernels, fpmp2_accuracy::high the Taylor kernel. Each result must lie
//  within 8u^2 (u = 2^-24) of an MPFR reference, held as a double pair, and
//  sin, cos, sinpi and cospi within 6u^2 at fpmp2_accuracy::high (over random
//  arguments the minimax kernels reach 6.0u^2 and 7.2u^2 for tan, the Taylor
//  kernel 4.2u^2 and 7.2u^2; on the cases below every level stays under
//  2.5u^2). Kernels that round their single-precision part into the pair steps
//  err by about 70u^2 near the ends of the reduced range. sin and cos must return
//  the values sincos returns, and sinpi and cospi those of sincospi.
//  Arguments next to a multiple of pi/2 are in math_reduction.pass.cpp.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

struct trig_case
{
  float x_hi;
  float x_lo;
  double sin_hi;
  double sin_lo;
  double cos_hi;
  double cos_lo;
  double tan_hi;
  double tan_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
template <class MP2>
TEST_HOST_DEVICE_FUNC double rel_err_u2(MP2 r, double ref_hi, double ref_lo)
{
  double d = ((static_cast<double>(r.hi()) - ref_hi) + static_cast<double>(r.lo())) - ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(ref_hi) * 0x1p48;
}

template <class MP2>
TEST_HOST_DEVICE_FUNC bool same_value(MP2 a, MP2 b)
{
  return a.hi() == b.hi() && a.lo() == b.lo();
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_accuracy(double sin_cos_bound, double tan_bound)
{
  // MPFR 4.2.2, 256-bit precision: x, sin(x), cos(x) and tan(x).
  trig_case cases[] = {
    {0x1.4p-20f,
     0x1.2p-46f,
     0x1.40000047ffacbp-20,
     -0x1.55636553b7d4bp-74,
     0x1.fffffffffe7p-1,
     -0x1.67ff97fdd4f79p-66,
     0x1.4000004800a6bp-20,
     -0x1.5539353b44569p-74},
    {-0x1.3ap-3f,
     0x1.7p-29f,
     -0x1.38c56f94d9214p-3,
     0x1.0f01b2f49b8acp-58,
     0x1.f9fe73fa56e87p-1,
     0x1.054536b02e5e6p-56,
     -0x1.3c7bd7e7d965dp-3,
     0x1.5ce2b0e283a9ap-58},
    {0x1.333334p-2f,
     -0x1.2e5d9ep-30f,
     0x1.2e9cda0d4501fp-2,
     -0x1.5aeffe1d77517p-60,
     0x1.e921dd2778908p-1,
     -0x1.d515e0d187909p-55,
     0x1.3cc2a519da799p-2,
     0x1.28a362fe6a6b5p-56},
    {0x1.91ae54p-1f,
     0x1.9a8566p-27f,
     0x1.69b9b1da4786ep-1,
     -0x1.ee4a645348171p-57,
     0x1.6a5a0934d13ddp-1,
     -0x1.1d1e6e378c17ap-58,
     0x1.ff1d706c960e1p-1,
     0x1.867a9a9761dbcp-60},
    {-0x1.915acep-1f,
     0x1.95d42ap-27f,
     -0x1.697e900bccfbbp-1,
     0x1.955539b376e52p-55,
     0x1.6a9507384ac77p-1,
     -0x1.8c3c6b6eaf426p-56,
     -0x1.fe76c7d29a0c3p-1,
     0x1.43d4da5d00f58p-55},
    {0x1.925856p-1f,
     0x0p+0f,
     0x1.6a31eeee0b3ep-1,
     0x1.8ab81eae10f9p-56,
     0x1.69e1d9741d85ap-1,
     0x1.235a39c42a76ep-56,
     0x1.0038a7000548p+0,
     -0x1.93003f5440badp-55},
    {0x1.333334p+0f,
     0x1.f2375ap-26f,
     0x1.dd343b10557afp-1,
     -0x1.c0141050e0403p-56,
     0x1.730de477d5751p-2,
     -0x1.ded6cfc09874p-58,
     0x1.493c4892de145p+1,
     -0x1.512d10d1be1c4p-53},
    {-0x1.4p+1f,
     0x1.ad7f2ap-24f,
     -0x1.326af38d2a02p-1,
     0x1.4fbdb71a9193bp-55,
     -0x1.9a2f7cf6434aap-1,
     0x1.fa3fe3a082a44p-56,
     0x1.7e79ba1a64d9ap-1,
     -0x1.b4fbae6ae3cd7p-55},
    {0x1.f69e7cp+1f,
     0x0p+0f,
     -0x1.69f003a7a2daap-1,
     0x1.4061fcd867721p-59,
     -0x1.6a23c74e9683ap-1,
     -0x1.f207150096c8ep-57,
     0x1.ffb6d090a93p-1,
     0x1.3d309e869fa59p-57},
    {0x1.5fd5d8p+2f,
     0x1.fbd664p-23f,
     -0x1.6a2b408c9d08ep-1,
     0x1.0af829176b181p-55,
     0x1.69e88930728abp-1,
     -0x1.3b90582794b4fp-55,
     -0x1.002f3147c6a91p+0,
     0x1.85044ef8ec692p-55},
    {-0x1.5fde2ap+2f,
     -0x1.a4472ep-25f,
     0x1.69fc352d86b14p-1,
     -0x1.043bef2267cecp-55,
     0x1.6a17971dd477cp-1,
     0x1.af4cd3057936ep-56,
     0x1.ffd947e9ed473p-1,
     -0x1.6bba209d8a23dp-55},
    {0x1.c45bd4p+2f,
     0x0p+0f,
     0x1.69dd8526148c7p-1,
     -0x1.a3b30925bb9dbp-58,
     0x1.6a364239ca16ep-1,
     0x1.7667db1338596p-55,
     0x1.ff8290920ef57p-1,
     0x1.4950541b92cafp-55},
    {0x1.aa785p+8f,
     -0x1.c651c4p-18f,
     -0x1.6a7ca476b467bp-1,
     0x1.de78ca99bd9ccp-55,
     0x1.699703f002d0dp-1,
     0x1.318bcadeb0815p-56,
     -0x1.00a2928692236p+0,
     -0x1.dd46a6ed05fb8p-55},
    {-0x1.ecd654p+9f,
     0x1.a15d4cp-20f,
     0x1.6a21d2708f22fp-1,
     -0x1.18b05dbb4ac2p-55,
     0x1.69f1f8ca97088p-1,
     -0x1.268fb303b2bbap-58,
     0x1.0021d80bd613dp+0,
     0x1.ef486b3301b2ep-54},
    {0x1.02d8d4p+16f,
     -0x1.3a41cp-10f,
     0x1.6a9677439e6dfp-1,
     0x1.66056b5a92974p-55,
     -0x1.697d1ee37eab8p-1,
     0x1.7f4357d764834p-55,
     -0x1.00c73e668d57ap+0,
     0x1.8132f513ff93cp-54},
    {0x1.32051cp+16f,
     0x1.f4608cp-9f,
     0x1.6919b6a2d2ab3p-1,
     0x1.a37a678d791d9p-56,
     -0x1.6af9773dad573p-1,
     0x1.cf50b24101f3bp-55,
     -0x1.fd5b46e364d28p-1,
     0x1.db57f50313d3bp-56},
    {-0x1.c342bep+19f,
     0x1.f3fd7ep-6f,
     -0x1.5bb186b6e17c1p-1,
     0x1.899b7ca9fe7edp-56,
     0x1.77d634b87b721p-1,
     -0x1.b2b9c2cf6dfd2p-55,
     -0x1.d9a91c57a4854p-1,
     0x1.a5259e052f90fp-55},
    {0x1.c5b482p+19f,
     -0x1.b5ed12p-6f,
     -0x1.77868a20a5be1p-1,
     0x1.024e2433c423ep-55,
     0x1.5c079054b0729p-1,
     -0x1.af76d4bbab1d1p-59,
     -0x1.1439a8534c1eap+0,
     0x1.3dddb9e58ce78p-57},
    {0x1.fff64p+19f,
     0x1.fffffep-6f,
     -0x1.93900e5874dcfp-1,
     -0x1.c00e0de8e5c93p-55,
     -0x1.3b16ce51c6df2p-1,
     -0x1.551a06a62aaeap-55,
     0x1.47e1ce2c2083ap+0,
     0x1.eaa60795751cp-55},
    {-0x1.fff64p+19f,
     -0x1.fffffep-6f,
     0x1.93900e5874dcfp-1,
     0x1.c00e0de8e5c93p-55,
     -0x1.3b16ce51c6df2p-1,
     -0x1.551a06a62aaeap-55,
     -0x1.47e1ce2c2083ap+0,
     -0x1.eaa60795751cp-55},
    {0x1.000002p+20f,
     0x1.8p-5f,
     0x1.f2bb7178e1bb4p-2,
     0x1.1d1833a1a7b43p-57,
     0x1.bf2b635ad15dep-1,
     -0x1.d024875dce5d9p-55,
     0x1.1d84e5d964da6p-1,
     -0x1.06fb4c85e2922p-55},
    {-0x1.cf2cb6p+21f,
     -0x1.bb310cp-4f,
     0x1.699bbfa84320ep-1,
     -0x1.11b7e588eac36p-60,
     0x1.6a77ebae440c5p-1,
     -0x1.b22dad56f8d9bp-59,
     0x1.fec8ffbe88dcfp-1,
     0x1.20befcbf9ad59p-55},
    {0x1.2a05f2p+33f,
     0x1.4p+8f,
     -0x1.a0f810e065bbp-1,
     0x1.3a97d0cb390b3p-55,
     0x1.291ef53d07debp-1,
     -0x1.b33b0d8165e1ep-56,
     -0x1.6742e7c26be1ap+0,
     0x1.f77e5ae394ab5p-54},
    {-0x1.93e594p+99f,
     0x1.1p+74f,
     0x1.fcb81f3163d15p-1,
     -0x1.2b856beee9b0bp-57,
     -0x1.cef26ffbaa3a6p-4,
     0x1.ccf81fc556845p-58,
     -0x1.194fa246e0976p+3,
     0x1.988c1cd91dd7ap-52},
    {0x1.fffffep+127f,
     0x0p+0f,
     -0x1.0b33665089575p-1,
     -0x1.06cee446be04ap-56,
     0x1.b4bf2c79bdfcep-1,
     -0x1.56b04a4ffb027p-55,
     -0x1.393d94b4988c2p-1,
     0x1.b579746c104c3p-58},
  };
  for (trig_case c : cases)
  {
    MP2 x(c.x_hi, c.x_lo);
    MP2 s  = cuda::sin(x);
    MP2 co = cuda::cos(x);
    assert(rel_err_u2(s, c.sin_hi, c.sin_lo) <= sin_cos_bound);
    assert(rel_err_u2(co, c.cos_hi, c.cos_lo) <= sin_cos_bound);
    assert(rel_err_u2(cuda::tan(x), c.tan_hi, c.tan_lo) <= tan_bound);
    MP2 s2, co2;
    cuda::sincos(x, &s2, &co2);
    assert(same_value(s, s2));
    assert(same_value(co, co2));
  }
}

struct pi_case
{
  float x_hi;
  float x_lo;
  double sin_hi;
  double sin_lo;
  double cos_hi;
  double cos_lo;
};

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_accuracy_pi(double sin_cos_bound)
{
  // MPFR 4.2.2, 256-bit precision: x, sinpi(x) and cospi(x). The fraction is taken from the
  // high limb below 2^23 and from the low limb from there on.
  pi_case cases[] = {
    {0x1.4p-20f, 0x1.2p-46f, 0x1.f6a7a3066764fp-19, 0x1.8a8f4e7787611p-74, 0x1.fffffffff0943p-1, -0x1.e123468901d5fp-55},
    {0x1.99999ap-4f,
     -0x1.6p-30f,
     0x1.3c6ef37dc02bp-2,
     0x1.13e46cdc67ac3p-56,
     0x1.e6f0e13285f71p-1,
     0x1.3741757abb2d4p-55},
    {-0x1.fp-3f, 0x1.8p-29f, -0x1.610b751b376d4p-1, -0x1.4d0c29f22b4cep-56, 0x1.72d083b2fd72bp-1, -0x1.79ab701177128p-57},
    {0x1.2p-1f, 0x1.cp-27f, 0x1.f6297cbad0e2ap-1, -0x1.170d6561e284cp-55, -0x1.8f8b892afe3bdp-3, -0x1.7d468aea9d81cp-57},
    {-0x1.d5p+0f, -0x1.4p-25f, 0x1.01cfc510347dp-1, 0x1.1cdf9eececdd4p-55, 0x1.ba5aa86d8f4f3p-1, -0x1.0f95b5342888ap-56},
    {0x1.e24p+9f,
     0x1.9p-16f,
     0x1.ffffffe7e77f5p-1,
     -0x1.0518854505ad2p-55,
     -0x1.3a28c598668dap-14,
     -0x1.7ea6e65c36dc2p-69},
    {0x1.6a09e6p+19f,
     -0x1.2p-6f,
     -0x1.0485626ae221ap-1,
     -0x1.b937d9091ff7p-55,
     -0x1.b8c38d27504e9p-1,
     0x1.1529abff40e45p-55},
    {-0x1.000002p+23f,
     0x1.4p-3f,
     -0x1.e2b5d3806f63bp-2,
     -0x1.e0d891d3c6841p-58,
     -0x1.c38b2f180bdb1p-1,
     0x1.6e0b1757c8d07p-56},
    {0x1.8p+30f, -0x1.3p-2f, -0x1.9b3e047f38741p-1, 0x1.30ee286712474p-55, 0x1.30ff7fce17035p-1, -0x1.efcc626f74a6fp-57},
  };
  for (pi_case c : cases)
  {
    MP2 x(c.x_hi, c.x_lo);
    MP2 s  = cuda::sinpi(x);
    MP2 co = cuda::cospi(x);
    assert(rel_err_u2(s, c.sin_hi, c.sin_lo) <= sin_cos_bound);
    assert(rel_err_u2(co, c.cos_hi, c.cos_lo) <= sin_cos_bound);
    MP2 s2, co2;
    cuda::sincospi(x, &s2, &co2);
    assert(same_value(s, s2));
    assert(same_value(co, co2));
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_zero(float zero)
{
  MP2 x(zero, 0.0f);
  MP2 s = cuda::sin(x);
  MP2 t = cuda::tan(x);
  MP2 s2, co2;
  cuda::sincos(x, &s2, &co2);
  // sin and tan return the zero they are given, sign included.
  assert(s.hi() == 0.0f && cuda::std::signbit(s.hi()) == cuda::std::signbit(zero));
  assert(t.hi() == 0.0f && cuda::std::signbit(t.hi()) == cuda::std::signbit(zero));
  assert(s2.hi() == 0.0f && cuda::std::signbit(s2.hi()) == cuda::std::signbit(zero));
  assert(s.lo() == 0.0f && t.lo() == 0.0f && s2.lo() == 0.0f);
  assert(same_value(cuda::cos(x), MP2(1.0f, 0.0f)));
  assert(same_value(co2, MP2(1.0f, 0.0f)));
}

// x^2 is below the smallest subnormal: sin(x) and tan(x) are x and cos(x) is 1, with a zero low limb.
template <class MP2>
TEST_HOST_DEVICE_FUNC void test_tiny(float tiny)
{
  MP2 x(tiny, 0.0f);
  MP2 s2, co2;
  cuda::sincos(x, &s2, &co2);
  assert(same_value(cuda::sin(x), x));
  assert(same_value(cuda::tan(x), x));
  assert(same_value(s2, x));
  assert(same_value(cuda::cos(x), MP2(1.0f, 0.0f)));
  assert(same_value(co2, MP2(1.0f, 0.0f)));
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_non_finite(float v)
{
  MP2 x(v, 0.0f);
  MP2 s2, co2;
  cuda::sincos(x, &s2, &co2);
  assert(cuda::std::isnan(cuda::sin(x).hi()));
  assert(cuda::std::isnan(cuda::cos(x).hi()));
  assert(cuda::std::isnan(cuda::tan(x).hi()));
  assert(cuda::std::isnan(s2.hi()));
  assert(cuda::std::isnan(co2.hi()));
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_special()
{
  test_zero<MP2>(0.0f);
  test_zero<MP2>(-0.0f);

  float denorm_min = cuda::std::numeric_limits<float>::denorm_min();
  float min        = cuda::std::numeric_limits<float>::min();
  float tiny[]     = {denorm_min, -denorm_min, 0x1.fffffcp-127f, min, -min, 0x1p-100f, -0x1.8p-80f};
  for (float v : tiny)
  {
    test_tiny<MP2>(v);
  }

  float inf = cuda::std::numeric_limits<float>::infinity();
  test_non_finite<MP2>(inf);
  test_non_finite<MP2>(-inf);
  test_non_finite<MP2>(cuda::std::numeric_limits<float>::quiet_NaN());
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test(double sin_cos_bound, double tan_bound)
{
  test_accuracy<MP2>(sin_cos_bound, tan_bound);
  test_accuracy_pi<MP2>(sin_cos_bound);
  test_special<MP2>();
}

int main(int, char**)
{
  test<cuda::fp32mp2>(8.0, 8.0);
  test<cuda::fp32mp2_low>(8.0, 8.0);
  test<cuda::fp32mp2_high>(6.0, 8.0);
  return 0;
}
