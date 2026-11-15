// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 erfc across its kernels: small arguments, where the result
//  is close to 1; the rest of the range up to |x| = 8.2, both signs, where
//  erfc(x) > 2^-101 and so the pair still resolves u^2 (u = 2^-24) of the
//  result; results in the subnormal range; and the arguments from which the
//  result is exactly 0 or 2. Up to |x| = 8.2 a result must lie within a bound of
//  an MPFR reference, held as a double pair: 16u^2 for the low and default
//  levels, which share one kernel, and 8u^2 for fpmp2_accuracy::high, which has
//  a more accurate one. A subnormal result must lie within 2^-148 at every level.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

struct erfc_case
{
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

// |r - ref|. r.hi() - ref_hi is exact, as the two are within a factor of 2 (or both
// subnormal), so the measurement itself errs by about 2^-53 of the reference.
template <class MP2>
TEST_HOST_DEVICE_FUNC double abs_err(MP2 r, erfc_case c)
{
  return cuda::std::fabs(((static_cast<double>(r.hi()) - c.ref_hi) + static_cast<double>(r.lo())) - c.ref_lo);
}

// bound_u2: the largest admissible relative error, in u^2 = 2^-48.
template <class MP2>
TEST_HOST_DEVICE_FUNC void test_erfc_normal(double bound_u2)
{
  // MPFR 4.2.2, 256-bit precision: x and erfc(x).
  erfc_case cases[] = {
    // |x| < 1/8
    {0x0p+0f, 0x0p+0f, 0x1p+0, 0x0p+0},
    {-0x0p+0f, 0x0p+0f, 0x1p+0, 0x0p+0},
    {0x1.0d9498p-8f, 0x1.b2246ap-34f, 0x1.fd9fa020eac8ep-1, -0x1.14c18584fb1e3p-55},
    {-0x1.17f05ep-10f, -0x1.d0105cp-39f, 0x1.004ef8231a1bfp+0, 0x1.e83f7dfd4fb3ap-56},
    {0x1.e2a844p-12f, 0x1.1ee37cp-37f, 0x1.ffbbec25b03ffp-1, 0x1.fa9a8992e7a83p-55},
    {-0x1.2af3b8p-9f, 0x1.4cdef2p-34f, 0x1.00a8aa527d3d8p+0, -0x1.8aaba13929d0ap-54},
    {0x1.fb59fp-11f, 0x0p+0f, 0x1.ff70e0fe4a611p-1, -0x1.f26d2b61b5d35p-61},
    {-0x1.0a9642p-12f, -0x1.000426p-39f, 0x1.0012ccfa7895cp+0, -0x1.f29488e0b52efp-59},
    // 1/8 <= |x| <= 8.2
    {0x1.9fed5ap-3f, 0x1.acd966p-29f, 0x1.8c4330e9ffe91p-1, 0x1.7fff6073ac9dcp-55},
    {0x1.e53acap-2f, -0x1.d0f138p-32f, 0x1.016b49d34df6ep-1, 0x1.c7cbea496fd57p-56},
    {0x1.a24ef4p-1f, -0x1.7f2248p-28f, 0x1.fbbbaec989835p-3, -0x1.88d919f34d513p-58},
    {0x1.516dbcp+0f, 0x1.c792c8p-30f, 0x1.fe7c1535b4701p-5, -0x1.2cac5765bccd1p-59},
    {0x1.1276fp+1f, 0x0p+0f, 0x1.3df7ba897ab77p-9, -0x1.a70bdabc8f773p-63},
    {0x1.b1dd52p+1f, -0x1.fde4f8p-24f, 0x1.b7ccee8df8fc3p-20, -0x1.d07fab495d5b6p-75},
    {0x1.03ac4p+2f, 0x1.a20782p-25f, 0x1.4924c4d68c8a3p-27, 0x1.26cc99e0bc94cp-81},
    {0x1.511a0cp+2f, 0x1.8514dp-23f, 0x1.a7abf5ef76314p-44, 0x1.25c8c686f9274p-98},
    {0x1.b11b62p+2f, -0x1.6b5546p-23f, 0x1.41d944f3e9d06p-70, -0x1.1ccefbcdc14d6p-124},
#if !TEST_HAS_CUDA_FAST_MATH // subnormal operand, result or low limb: flushed to zero on the device
    {0x1.0308dep+3f, 0x0p+0f, 0x1.8740de772fc47p-99, 0x1.7ebd00bf695ap-154},
    {0x1.04fd0ep+3f, -0x1.134a66p-23f, 0x1.1ff8c7a1cbb05p-100, -0x1.eef60360bdca8p-154},
#endif // !TEST_HAS_CUDA_FAST_MATH
    {-0x1.374342p-1f, 0x1.9bd734p-26f, 0x1.9c2dbfda0b7fdp+0, 0x1.51ec7e5421fe7p-54},
    {-0x1.bfc7fap+0f, 0x1.765ea8p-29f, 0x1.fc938db6e6b8dp+0, 0x1.5a93018b424e5p-54},
    {-0x1.ff5ecep+1f, -0x1.5bbc9ap-25f, 0x1.ffffffbb0c0ap+0, 0x1.9e3c3302ee41bp-54},
    {-0x1.63bc46p+2f, 0x0p+0f, 0x1.fffffffffffefp+0, -0x1.9bd274384a06fp-55},
  };
  for (erfc_case c : cases)
  {
    MP2 r = cuda::erfc(MP2(c.x_hi, c.x_lo));
    assert(abs_err(r, c) <= bound_u2 * 0x1p-48 * cuda::std::fabs(c.ref_hi));
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_erfc_subnormal()
{
  // erfc(x) < 2^-126 from x = 9.1946 on, and erfc(x) < 2^-150 from x = 10.0542 on.
  erfc_case cases[] = {
    {0x1.266f6ap+3f, 0x1.5758d8p-25f, 0x1.c58d86cf923c9p-127, -0x1.af4337dbb003ep-181},
    {0x1.2cd356p+3f, 0x1.42dd62p-22f, 0x1.5a2554f3ec144p-132, -0x1.d2df96cdd61b3p-187},
    {0x1.376e24p+3f, 0x1.d53a34p-22f, 0x1.2df5476dd6544p-141, 0x1.e4db529df65ap-196},
    {0x1.3e7054p+3f, -0x1.563fbep-24f, 0x1.fb66033acc9e9p-148, -0x1.b3ee6cc73f445p-202},
    {0x1.4163ep+3f, -0x1.5388dep-25f, 0x1.3e186581fbda9p-150, -0x1.46d88e04e2682p-204},
    {0x1.41cp+3f, 0x0p+0f, 0x1.faedd62dabd96p-151, -0x1.fa59183c89496p-207},
  };
  for (erfc_case c : cases)
  {
    MP2 r = cuda::erfc(MP2(c.x_hi, c.x_lo));
    assert(abs_err(r, c) <= 0x1p-148);
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test_erfc_saturated()
{
  // From the float above 10.0546875 on the result is exactly 0, and exactly 2 for -x.
  float inf  = cuda::std::numeric_limits<float>::infinity();
  float xs[] = {0x1.41c002p+3f, 10.5f, 27.5f, 28.0f, 1e10f, cuda::std::numeric_limits<float>::max(), inf};
  for (float x : xs)
  {
    MP2 p = cuda::erfc(MP2(x));
    assert(p.hi() == 0.0f && p.lo() == 0.0f);
    MP2 m = cuda::erfc(MP2(-x));
    assert(m.hi() == 2.0f && m.lo() == 0.0f);
  }
}

template <class MP2>
TEST_HOST_DEVICE_FUNC void test(double bound_u2)
{
  test_erfc_normal<MP2>(bound_u2);
#if !TEST_HAS_CUDA_FAST_MATH
  // subnormal results: --use_fast_math flushes them to zero on the device (FTZ)
  test_erfc_subnormal<MP2>();
#endif // !TEST_HAS_CUDA_FAST_MATH
  test_erfc_saturated<MP2>();
}

int main(int, char**)
{
  test<cuda::fp32mp2>(16.0);
  test<cuda::fp32mp2_low>(16.0);
  test<cuda::fp32mp2_high>(8.0);
  return 0;
}
