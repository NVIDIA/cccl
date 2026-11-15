// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 erfcx and normcdf across their kernels. erfcx: small
//  arguments, the range of the one-reciprocal kernel (up to 10.0546875), the
//  range beyond it up to 40, the asymptotic series, and negative arguments down
//  to the overflow threshold. normcdf: small arguments, both tails while the
//  result is above 2^-101, so that the pair still resolves u^2 (u = 2^-24) of
//  it, results in the subnormal range, and the arguments from which the result
//  is exactly 0 or 1. A result must lie within a bound of an MPFR reference,
//  held as a double pair: 16u^2 for the low and default levels, which share the
//  one-reciprocal kernel with erfc, and 8u^2 for fpmp2_accuracy::high, which
//  keeps the kernel with the two refined divisions. A subnormal result must lie
//  within 2^-148 at every level.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

struct ref_case
{
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

// |r - ref|. r.hi() - ref_hi is exact, as the two are within a factor of 2 (or both
// subnormal), so the measurement itself errs by about 2^-53 of the reference.
template <class MP2>
TEST_HOST_DEVICE_FUNC double abs_err(MP2 r, ref_case c)
{
  return cuda::std::fabs(((static_cast<double>(r.hi()) - c.ref_hi) + static_cast<double>(r.lo())) - c.ref_lo);
}

// bound_u2: the largest admissible relative error, in u^2 = 2^-48.
template <class MP2>
TEST_HOST_DEVICE_FUNC void test_erfcx(double bound_u2)
{
  // MPFR 4.2.2, 1024-bit precision: x and erfcx(x) = exp(x^2) erfc(x).
  ref_case cases[] = {
    // |x| < 1/2
    {0x0p+0f, 0x0p+0f, 0x1p+0, 0x0p+0},
    {0x1.86ad14p-34f, 0x1.a88618p-59f, 0x1.ffffffff2395bp-1, -0x1.ad94d00f363bdp-55},
    {-0x1.1a7932p-8f, 0x1.bf4e98p-33f, 0x1.013ff565afe43p+0, -0x1.e31e9edd947e5p-57},
    {-0x1.028e96p-2f, 0x1.ea7772p-29f, 0x1.5cf88d180438ep+0, -0x1.bddb48f0fd0aep-54},
    {0x1.90f156p-2f, -0x1.008cdap-28f, 0x1.5a05526236703p-1, -0x1.9c8df47d86c38p-56},
    {0x1.10ed0ep-4f, 0x0p+0f, 0x1.dbab2a997e43p-1, 0x1.77331f34019e2p-56},
    // 1/2 <= x <= 10.0546875
    {0x1.0e68dcp-1f, 0x1.7e70eap-26f, 0x1.33fd31c9c20d5p-1, -0x1.0943027cdc7a8p-55},
    {0x1.c54daap-1f, 0x1.2fde6p-29f, 0x1.d82028f67e8cp-2, -0x1.7cda2ee231bb7p-57},
    {0x1.9cefeap+0f, 0x1.7458e4p-27f, 0x1.375069b171ba3p-2, -0x1.a4b5ae3106cafp-56},
    {0x1.5f8f2ep+1f, 0x0p+0f, 0x1.8d10fce728f9fp-3, -0x1.963547ebc457ap-57},
    {0x1.ae801ap+1f, -0x1.f9cdbcp-24f, 0x1.4a08c9f92e00ep-3, -0x1.c55bb25e0805cp-58},
    {0x1.4245ccp+2f, -0x1.fdb28p-26f, 0x1.c25d353ee025p-4, 0x1.502bae7a5412ap-59},
    {0x1.733712p+2f, 0x1.87848p-23f, 0x1.88be439a01c4fp-4, -0x1.b4958b8a3bba1p-58},
    {0x1.c123f4p+2f, 0x1.485ca6p-23f, 0x1.460c1111fe748p-4, -0x1.ef26dce430f0dp-59},
    {0x1.1ea272p+3f, -0x1.51bc78p-22f, 0x1.0069ddc7563d2p-4, -0x1.8b4cac41782d9p-59},
    {0x1.41cp+3f, 0x0p+0f, 0x1.c96e06d770949p-5, 0x1.f735c7882ab17p-59},
    // 10.0546875 < x <= 40
    {0x1.41c002p+3f, 0x0p+0f, 0x1.c96e04068fa9fp-5, 0x1.36a366151f51dp-59},
    {0x1.bb6424p+3f, -0x1.c263ccp-23f, 0x1.4cb37819eefecp-5, -0x1.9af36f4a328b2p-61},
    {0x1.68e654p+4f, 0x1.7ebb4ep-24f, 0x1.996791e945123p-6, 0x1.f28c0275c16dbp-60},
    {0x1.3ea828p+5f, 0x1.536a6ep-20f, 0x1.cffc6f26abc8ep-7, 0x1.caf1a1aed6c09p-63},
    // x > 40: the asymptotic series
    {0x1.4b1302p+5f, -0x1.48062ap-23f, 0x1.be981097d8b89p-7, 0x1.faf93262a49b2p-61},
    {0x1.e54542p+9f, -0x1.51a736p-16f, 0x1.30c6ab5575ec7p-11, 0x1.5959a284925aep-65},
    {0x1.9254d4p+34f, -0x1p+9f, 0x1.6f9ad093698bap-36, -0x1.adc07d7d2599cp-90},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {0x1.93e594p+99f, 0x0p+0f, 0x1.6e2e127fc5d5fp-101, 0x1.adfe11580a1abp-157},
#endif // !TEST_HAS_CUDA_FAST_MATH
    // -9.3824 <= x <= -1/2
    {-0x1.415bccp-1f, -0x1.a7755p-28f, 0x1.347a412c11217p+1, -0x1.2da085751f944p-53},
    {-0x1.bd38acp+0f, 0x1.bd43c8p-25f, 0x1.47172c499abeap+5, -0x1.8bb536fecd023p-49},
    {-0x1.9c8aa6p+1f, 0x1.6a05dap-24f, 0x1.fb1b597fbed32p+15, -0x1.fbcb98ca76de6p-39},
    {-0x1.dcc74cp+2f, 0x0p+0f, 0x1.0bf9ecf735611p+81, 0x1.f3cfe5607d80ap+25},
    {-0x1.21ccb2p+3f, 0x1.72a4b4p-23f, 0x1.404b499c017f4p+119, 0x1.f453b5f23f1ffp+65},
  };
  for (ref_case c : cases)
  {
    MP2 r = cuda::erfcx(MP2(c.x_hi, c.x_lo));
    assert(abs_err(r, c) <= bound_u2 * 0x1p-48 * cuda::std::fabs(c.ref_hi));
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_erfcx_ends()
{
  // 2 exp(x^2) passes FLT_MAX below -9.3824; erfcx(+inf) = 0.
  float inf  = cuda::std::numeric_limits<float>::infinity();
  float xs[] = {-9.39f, -10.5f, -40.0f, -1e10f, -inf};
  for (float x : xs)
  {
    MP2 r = cuda::erfcx(MP2(x));
    assert(r.hi() == inf && r.lo() == 0.0f);
  }
  MP2 z = cuda::erfcx(MP2(inf));
  assert(z.hi() == 0.0f && z.lo() == 0.0f);
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_normcdf(double bound_u2)
{
  // MPFR 4.2.2, 1024-bit precision: x and normcdf(x) = erfc(-x / sqrt(2)) / 2.
  ref_case cases[] = {
    // |x| < sqrt(1/2): the small erf
    {0x0p+0f, 0x0p+0f, 0x1p-1, 0x0p+0},
    {0x1.86ad14p-34f, 0x1.a88618p-59f, 0x1.000000004dedcp-1, -0x1.b43aded04a70bp-55},
    {-0x1.1a7932p-8f, 0x1.bf4e98p-33f, 0x1.fe3d3d32d618fp-2, 0x1.7a0b5dd8650e6p-58},
    {-0x1.50a2aep-2f, 0x1.1e5332p-27f, 0x1.7c1509d718fd4p-2, -0x1.b0317c8b07e9ap-58},
    {0x1.131b66p-1f, 0x1.6c97fap-26f, 0x1.68b11973f981dp-1, 0x1.10aa948196a02p-55},
    {0x1.10ed0ep-4f, 0x0p+0f, 0x1.0d99a3731fdbap-1, -0x1.8fedbbfd55cb5p-55},
    // -11.5 <= x <= -sqrt(1/2): normcdf(x) > 2^-101
    {-0x1.8ca1ap-1f, 0x1.16cb4p-27f, 0x1.c10f24d97bdecp-3, -0x1.756d6352a6e0ap-59},
    {-0x1.0028d8p+0f, 0x1.e8ae4p-31f, 0x1.449e01afd19d7p-3, -0x1.652d535688e83p-58},
    {-0x1.0a4c1p+1f, -0x1.745eacp-25f, 0x1.3312f0c7a139cp-6, 0x1.5dd6eef5d4becp-60},
    {-0x1.c0e1a6p+1f, 0x0p+0f, 0x1.db680d3ffaabfp-13, 0x1.ae1437a2e4505p-67},
    {-0x1.917fe6p+2f, -0x1.f9cdbcp-23f, 0x1.845247ccd1febp-33, -0x1.9111c62f90295p-91},
    {-0x1.e7a2fp+2f, -0x1.53cc56p-25f, 0x1.cb65346fdada4p-47, -0x1.6ca804e9c7ab1p-101},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {-0x1.466476p+3f, -0x1.3c3dcp-22f, 0x1.338f3ebbe8f4fp-80, 0x1.6d09c0f5c518cp-135},
    {-0x1.6f9eaep+3f, -0x1.cfc23cp-23f, 0x1.eb391219c6631p-101, -0x1.284f3775c5686p-156},
#endif // !TEST_HAS_CUDA_FAST_MATH
    // sqrt(1/2) <= x < 8.4
    {0x1.7991b4p-1f, 0x1.f26d3p-27f, 0x1.8a0568c9b9451p-1, -0x1.82a183bed3e47p-57},
    {0x1.b846cep+0f, -0x1.fa3baep-26f, 0x1.ea1f1411404d5p-1, 0x1.d1ee2f9898df7p-56},
    {0x1.a1331ap+1f, 0x0p+0f, 0x1.ffb6d2b3d1fbdp-1, 0x1.d1e895bd4de56p-56},
    {0x1.7e58dp+2f, -0x1.a8d2a4p-24f, 0x1.fffffff6113a8p-1, 0x1.4f6c0353d1c88p-57},
    {0x1.aa4d72p+2f, -0x1.b06cb6p-25f, 0x1.ffffffffe2178p-1, 0x1.817dc688f40ffp-62},
  };
  for (ref_case c : cases)
  {
    MP2 r = cuda::normcdf(MP2(c.x_hi, c.x_lo));
    assert(abs_err(r, c) <= bound_u2 * 0x1p-48 * cuda::std::fabs(c.ref_hi));
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_normcdf_subnormal()
{
  // normcdf(x) < 2^-126 from x = -13.06 down, and normcdf(x) < 2^-150 from x = -14.17 down.
  ref_case cases[] = {
    {-0x1.a4bb6ep+3f, -0x1.8955dp-26f, 0x1.30eab76018142p-130, 0x1.777673d65cf26p-184},
    {-0x1.ae265p+3f, 0x1.8feb6ep-22f, 0x1.7d96d6ef46691p-136, -0x1.ad5a7d484a81ap-193},
    {-0x1.b8ac26p+3f, -0x1.12fb22p-23f, 0x1.0fc4a3e757399p-142, -0x1.c514284f97d58p-196},
    {-0x1.c2385cp+3f, -0x1.d0af5p-25f, 0x1.0b8ccc30ba23ap-148, 0x1.df2a746f8d427p-205},
    {-0x1.c66666p+3f, 0x0p+0f, 0x1.4ebae9fcf5c6fp-151, -0x1.9e2eb158f9e56p-206},
    {-0x1.c7p+3f, 0x0p+0f, 0x1.001affd862012p-151, -0x1.fd7b09e550204p-205},
  };
  for (ref_case c : cases)
  {
    MP2 r = cuda::normcdf(MP2(c.x_hi, c.x_lo));
    assert(abs_err(r, c) <= 0x1p-148);
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_normcdf_saturated()
{
  // Exactly 0 below -14.5 at every level, and exactly 1 from 8.4 on.
  float inf  = cuda::std::numeric_limits<float>::infinity();
  float lo[] = {-14.51f, -15.0f, -40.0f, -1e10f, -cuda::std::numeric_limits<float>::max(), -inf};
  for (float x : lo)
  {
    MP2 r = cuda::normcdf(MP2(x));
    assert(r.hi() == 0.0f && r.lo() == 0.0f);
  }
  float hi[] = {8.4f, 9.0f, 40.0f, 1e10f, cuda::std::numeric_limits<float>::max(), inf};
  for (float x : hi)
  {
    MP2 r = cuda::normcdf(MP2(x));
    assert(r.hi() == 1.0f && r.lo() == 0.0f);
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test(double bound_u2)
{
  test_erfcx<MP2>(bound_u2);
  test_erfcx_ends<MP2>();
  test_normcdf<MP2>(bound_u2);
#if !TEST_HAS_CUDA_FAST_MATH
  // subnormal results: --use_fast_math flushes them to zero on the device (FTZ)
  test_normcdf_subnormal<MP2>();
#endif // !TEST_HAS_CUDA_FAST_MATH
  test_normcdf_saturated<MP2>();
}

int main(int, char**)
{
  test<cuda::fp32mp2>(16.0);
  test<cuda::fp32mp2_low>(16.0);
  test<cuda::fp32mp2_high>(8.0);
  return 0;
}
