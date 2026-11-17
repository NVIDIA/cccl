// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 sin, cos, sincos and tan near a multiple of pi/2, and erfc
//  at large x, where the reduced argument is far smaller than the terms it is
//  computed from. Each result must lie within 32u^2 (u = 2^-24) of a binary128
//  reference, held as a double pair; the earlier reductions erred by more than
//  10^9 u^2 on the trig cases and by about 100u^2 on the erfc cases.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

struct math_case
{
  float x_hi;
  float x_lo;
  double ref_hi;
  double ref_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
TEST_HOST_DEVICE_FUNC double rel_err_u2(cuda::fp32mp2 r, math_case c)
{
  double d = ((static_cast<double>(r.hi()) - c.ref_hi) + static_cast<double>(r.lo())) - c.ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(c.ref_hi) * 0x1p48;
}

TEST_HOST_DEVICE_FUNC bool accurate(cuda::fp32mp2 r, math_case c)
{
  return rel_err_u2(r, c) <= 32.0;
}

TEST_HOST_DEVICE_FUNC void test_sin()
{
  math_case cases[] = {
    {0x1.b751ep+16f, 0x1.a9937cp-12f, 0x1.b704a3e87a088p-40, 0x1.7903896fcaaa7p-95},
    {-0x1.c3817ap+16f, -0x1.6d4c62p-12f, -0x1.c7813e02e9f6p-41, -0x1.b6c9ceeae7c08p-96},
    {0x1.75ba1p+16f, 0x1.699932p-13f, -0x1.778b444077fbap-41, 0x1.6e67cc3f7ac0bp-95},
    {0x1.2faddap+17f, 0x1.fe18bcp-12f, -0x1.93035442c703ep-39, 0x1.e3066e605e8c3p-93},
  };
  for (math_case c : cases)
  {
    cuda::fp32mp2 x(c.x_hi, c.x_lo);
    assert(accurate(cuda::sin(x), c));
    cuda::fp32mp2 s, co;
    cuda::sincos(x, &s, &co);
    assert(accurate(s, c));
  }
}

TEST_HOST_DEVICE_FUNC void test_cos()
{
  math_case cases[] = {
    {0x1.c1b6cep+14f, -0x1.3ed164p-14f, -0x1.4109b800cfd1bp-37, -0x1.80a623a54965p-91},
    {0x1.943436p+16f, -0x1.7d6794p-9f, 0x1.20f590ac00068p-43, 0x1.fda57022a4993p-98},
    {0x1.b6c2eep+16f, 0x1.aa53e8p-9f, -0x1.3cf320036ff95p-40, -0x1.5562a5aca920cp-94},
    {-0x1.82ecd8p+16f, -0x1.dd755cp-10f, 0x1.a94f3643effbcp-41, 0x1.b4f1aabb42d6cp-95},
    {0x1.2faddap+14f, 0x1.fe18bcp-15f, 0x1.93035442c703ep-42, -0x1.e3066dbc81125p-96},
  };
  for (math_case c : cases)
  {
    cuda::fp32mp2 x(c.x_hi, c.x_lo);
    assert(accurate(cuda::cos(x), c));
    cuda::fp32mp2 s, co;
    cuda::sincos(x, &s, &co);
    assert(accurate(co, c));
  }
}

TEST_HOST_DEVICE_FUNC void test_tan()
{
  math_case cases[] = {
    {0x1.8fe25ep+16f, 0x1.da0fc8p-13f, -0x1.984e3e5639da4p+41, 0x1.7c9d9765184c7p-13},
    {0x1.c5aa66p+14f, -0x1.3fb578p-12f, 0x1.9b5f03bfcbf6cp-43, -0x1.4d0ae04a1e53p-99},
    {0x1.543fccp+15f, 0x1.1037e6p-11f, 0x1.348742cfd8f91p-42, -0x1.f3905064cde4ap-99},
    {-0x1.d9cc2ep+16f, -0x1.2427e6p-12f, 0x1.4b4b4d9857139p+39, -0x1.70066bd2c02dbp-19},
  };
  for (math_case c : cases)
  {
    assert(accurate(cuda::tan(cuda::fp32mp2(c.x_hi, c.x_lo)), c));
  }
}

TEST_HOST_DEVICE_FUNC void test_erfc()
{
  math_case cases[] = {
    {0x1.a609a2p+2f, -0x1.de8b2p-24f, 0x1.a020042a7adcep-67, -0x1.10cb1ddcbb97ap-124},
    {0x1.a7ed96p+2f, -0x1.b3674p-25f, 0x1.186315fbf609ep-67, -0x1.258fda7581c6p-121},
    {0x1.6db216p+2f, -0x1.4f8d8p-23f, 0x1.72e36e3c9158bp-51, 0x1.879f0c9fe8df7p-110},
    {0x1.ae798p+2f, 0x1.bd2c4p-23f, 0x1.19fee8a65a0d4p-69, -0x1.41a806be42477p-123},
    {0x1.7f6346p+2f, 0x1.a150c8p-23f, 0x1.bde7f881aa76p-56, -0x1.00d44b0036a79p-111},
  };
  for (math_case c : cases)
  {
    assert(accurate(cuda::erfc(cuda::fp32mp2(c.x_hi, c.x_lo)), c));
  }
}

int main(int, char**)
{
  test_sin();
  test_cos();
  test_tan();
  test_erfc();

  return 0;
}
