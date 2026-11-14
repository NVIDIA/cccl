// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 cbrt and rcbrt across the argument reduction: every
//  residue of the exponent modulo 3, both signs, the ends of the normal range
//  and subnormal high limbs. The host and the device seed the Halley step
//  differently (a refined polynomial and the SFU lg2/ex2 pair); either way each
//  result must lie within 16u^2 (u = 2^-24) of an MPFR reference, held as a
//  double pair.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

struct cbrt_case
{
  float x_hi;
  float x_lo;
  double cbrt_hi;
  double cbrt_lo;
  double rcbrt_hi;
  double rcbrt_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
TEST_HOST_DEVICE_FUNC double rel_err_u2(cuda::fp32mp2 r, double ref_hi, double ref_lo)
{
  double d = ((static_cast<double>(r.hi()) - ref_hi) + static_cast<double>(r.lo())) - ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(ref_hi) * 0x1p48;
}

TEST_HOST_DEVICE_FUNC void test_cbrt()
{
  // MPFR 4.2.2, 256-bit precision: x, cbrt(x) and 1 / cbrt(x).
  cbrt_case cases[] = {
    {0x1.3c1ebap-2f,
     -0x1.9199f8p-28f,
     0x1.5a0915d3844ep-1,
     -0x1.992c0d90b6c97p-56,
     0x1.7ac82e5df444p+0,
     -0x1.4d37bc18c89d6p-54},
    {-0x1.533ep-1f,
     0x1.cf1818p-26f,
     -0x1.be5ba12e46e15p-1,
     0x1.7f4b40a6dec05p-56,
     -0x1.25a5d2c0b26e1p+0,
     0x1.750451cf1de44p-56},
    {0x1.eefa3p+0f,
     0x1.63d5cp-26f,
     0x1.3eecc51be0c31p+0,
     0x1.b3459fafc69cep-54,
     0x1.9afb150224aa3p-1,
     0x1.f788f8957a15dp-55},
    {-0x1.9a9892p+1f,
     0x0p+0f,
     -0x1.798ce3759c11dp+0,
     0x1.ea848fb21c871p-56,
     -0x1.5b2a0f4dfbddap-1,
     0x1.979ed69f05accp-55},
    {0x1.c8e28ap+2f,
     -0x1.a7eee8p-23f,
     0x1.eced34dbbb50ep+0,
     -0x1.d92e7caa94ed6p-54,
     0x1.09e7dd5e09eabp-1,
     -0x1.50a6ad53b1a05p-55},
    {-0x1.ee73b2p+3f,
     0x1.7f4008p-22f,
     -0x1.3ecfdf9582da1p+1,
     0x1.c49f8f651ca61p-53,
     -0x1.9b205529c07a4p-2,
     0x1.270a26a8409c6p-58},
    {0x1.73107p+127f,
     0x1.e587a8p+102f,
     0x1.6d05c478885a9p+42,
     0x1.a142fc2a438d1p-13,
     0x1.671446dad8861p-43,
     0x1.e88c76fdb9f84p-97},
    {-0x1.e637bap+126f,
     0x0p+0f,
     -0x1.3d08471f2f756p+42,
     0x1.4dbd62b1e6c65p-14,
     -0x1.9d6f25afc0f88p-43,
     0x1.62df59ca05923p-99},
    {0x1.eeca0cp-126f,
     0x0p+0f,
     0x1.3ee26dd2be81cp-42,
     -0x1.956e9e749aaf3p-96,
     0x1.9b0868d9bfe0ep+41,
     0x1.e7c6e99048aa9p-14},
    {-0x1.927f24p-125f,
     0x0p+0f,
     -0x1.770d25146e965p-42,
     -0x1.1f941548cbbdbp-96,
     -0x1.5d7a3c316ap+41,
     -0x1.31398dd733737p-13},
    {0x1.499c8p-128f,
     0x0p+0f,
     0x1.5ee3c6d24e20cp-43,
     0x1.c940875e1050bp-100,
     0x1.758ab5513e206p+42,
     0x1.ef50f0371dfbcp-13},
    {-0x1.f56348p-128f,
     0x0p+0f,
     -0x1.938c23475d4bp-43,
     0x1.77e05f390229dp-97,
     -0x1.44ccac5870666p+42,
     0x1.4c1b7e64450ap-18},
    {0x1.103268p-37f,
     -0x1.482748p-63f,
     0x1.9ec535360affdp-13,
     0x1.59da9799791dp-68,
     0x1.3c02d2549f04ep+12,
     -0x1.ae55088286c4fp-42},
    {-0x1.2f6132p+33f,
     -0x1.5dfcb8p+8f,
     -0x1.0ee87f5815155p+11,
     -0x1.16f4a227d58f2p-43,
     -0x1.e3d30d551796p-12,
     -0x1.0722142cdf829p-70},
    {0x1.a05954p+0f,
     -0x1.660e04p-25f,
     0x1.2d0e122ac5dcp+0,
     0x1.0c9837b436326p-54,
     0x1.b3602a6482568p-1,
     -0x1.f967f5d73719ep-55},
    {-0x1.acd664p+0f,
     0x0p+0f,
     -0x1.300912e03fc09p+0,
     -0x1.249c544d55f6cp-57,
     -0x1.af1b8dadd8ce7p-1,
     0x1.2e4c170e686ecp-55},
  };
  for (cbrt_case c : cases)
  {
    cuda::fp32mp2 x(c.x_hi, c.x_lo);
    assert(rel_err_u2(cuda::cbrt(x), c.cbrt_hi, c.cbrt_lo) <= 16.0);
    assert(rel_err_u2(cuda::rcbrt(x), c.rcbrt_hi, c.rcbrt_lo) <= 16.0);
  }
}

int main(int, char**)
{
  test_cbrt();
  return 0;
}
