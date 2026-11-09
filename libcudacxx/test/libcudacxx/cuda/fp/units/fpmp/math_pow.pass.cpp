// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

// fp32mp2 pow: within 4u^2 (u = 2^-24) of binary128 where b ln|a| is large, which magnifies an
// error in ln|a| (forming ln|a| to the format's own precision erred by up to 160u^2 on these
// cases), and for subnormal and tiny bases; the C99 Annex F special values on the value hi + lo;
// and gradual underflow below FLT_MIN.

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "test_macros.h"

struct pow_case
{
  float a_hi;
  float a_lo;
  float b_hi;
  float b_lo;
  double ref_hi;
  double ref_lo;
};

// Relative error of r in units of u^2 = 2^-48. r.hi() - ref_hi is exact, as the two are
// within a factor of 2, so the measurement itself errs by about 2^-53 of the reference.
TEST_HOST_DEVICE_FUNC double rel_err_u2(cuda::fp32mp2 r, pow_case c)
{
  double d = ((static_cast<double>(r.hi()) - c.ref_hi) + static_cast<double>(r.lo())) - c.ref_lo;
  return cuda::std::fabs(d) / cuda::std::fabs(c.ref_hi) * 0x1p48;
}

TEST_HOST_DEVICE_FUNC void test_accuracy()
{
  pow_case cases[] = {
    // a near 1 and b ln a near 88
    {0x1.01218ep+0f, -0x1.017fccp-25f, 0x1.2b09d6p+14f, -0x1.4a8608p-11f, 0x1.a677c77d4ef02p+121, -0x1.94b110743d4ebp+67},
    {0x1.0119dap+0f, -0x1.73045ap-25f, 0x1.3a41e4p+14f, -0x1.6a3f5p-11f, 0x1.6f71d2d935a9bp+124, 0x1.e1bb12cccce9ep+68},
    {0x1.fb991cp-1f, -0x1.ec616p-26f, 0x1.ce55f6p+10f, 0x1.77e33ep-17f, 0x1.f2cbe3022814p-24, 0x1.fea49a3221ccap-80},
    // a = 1 + a lo limb only, with a huge b
    {0x1p+0f, 0x1.5f8ff8p-35f, 0x1.d954a2p+40f, 0x1.03f032p+14f, 0x1.2ab8b5fe11aafp+117, 0x1.7315983379104p+59},
    {0x1p+0f, -0x1.4f90eep-26f, -0x1.86be18p+21f, -0x1.e1f6ep-5f, 0x1.10844dfd9e0a9p+0, -0x1.3a66ebc33dc76p-54},
    // a = 2^k plus a small lo limb
    {0x1p+94f, -0x1.595b5p+56f, 0x1.5795ap+0f, -0x1.125b46p-26f, 0x1.1e0292e69b5f6p+126, 0x1.a1e15310e6e47p+72},
    {0x1.c6621ep+14f, -0x1.60c98ep-11f, -0x1.36144p+2f, -0x1.8b028p-31f, 0x1.1df66a95c9c17p-72, 0x1.714e2aec29569p-126},
    // e^88.72, just below the overflow threshold, and e^-70, where the lo limb is still normal
    {0x1.5bf0a8p+1f, 0x1.628aeep-24f, 0x1.62e42ep+6f, 0x1.4p-20f, 0x1.ffff302e5aeap+127, 0x1.988c6796467b5p+72},
    {0x1.5bf0a8p+1f, 0x1.628aeep-24f, -0x1.18p+6f, 0.0f, 0x1.02057d1245b9p-101, 0x1.f6fbab6e10f2dp-155},
    {0x1.8p+0f, 0x1.2p-30f, -0x1.2345p+7f, 0x1.2p-18f, 0x1.c08cda748b05ep-86, 0x1.c8a635b14fa9ap-141},
    // negative bases with integer exponents, the last two integers only through b_lo
    {-0x1.375fb4p-4f, 0x1.6738bcp-30f, -0x1.bp+4f, 0.0f, -0x1.4b6452bb340bcp+100, 0x1.4e0ccd5a1bce9p+45},
    {-0x1p+0f, 0x1p-30f, 0x1p+30f, 0x1p+0f, -0x1.78b56359fab33p-2, 0x1.f7c41029d2c0fp-56},
    {-0x1p+0f, 0x1p-30f, -0x1p+31f, 0x1p+1f, 0x1.d8e64b85ea41bp+2, -0x1.6d101f078d62cp-54},
    {0x1.6a09e6p-1f, 0x1.9fcef4p-27f, 0x1.4p+3f, 0.0f, 0x1.0000000000018p-5, 0x1.2ac7d7f7f5dbp-59},
    // a subnormal base, and a tiny base whose lo limb is subnormal
    {0x1.2d687p-129f, 0.0f, -0x1.333334p-1f, 0.0f, 0x1.32450b3fe3e8cp+77, 0x1.567bef45e3887p+21},
    {0x1.3c5a7ep-118f, -0x1.98p-143f, 0x1.7ae148p-2f, 0.0f, 0x1.5e6edd1560973p-44, 0x1.51a4a2b5d3bebp-98},
  };
  for (pow_case c : cases)
  {
    const cuda::fp32mp2 r = cuda::pow(cuda::fp32mp2(c.a_hi, c.a_lo), cuda::fp32mp2(c.b_hi, c.b_lo));
    assert(rel_err_u2(r, c) <= 4.0);
  }
}

struct pow_exact_case
{
  float a_hi;
  float a_lo;
  float b_hi;
  float b_lo;
  float r; // expected hi limb, compared bit for bit (NaN matches any NaN)
};

TEST_HOST_DEVICE_FUNC bool same(float x, float y)
{
  if (cuda::std::isnan(y))
  {
    return cuda::std::isnan(x);
  }
  return x == y && cuda::std::signbit(x) == cuda::std::signbit(y);
}

template <int N>
TEST_HOST_DEVICE_FUNC void check_exact(const pow_exact_case (&cases)[N])
{
  for (const pow_exact_case& c : cases)
  {
    const cuda::fp32mp2 r = cuda::pow(cuda::fp32mp2(c.a_hi, c.a_lo), cuda::fp32mp2(c.b_hi, c.b_lo));
    assert(same(r.hi(), c.r));
    // A zero, infinite or subnormal result carries no lo limb.
    if (!cuda::std::isnan(c.r) && !cuda::std::isnormal(c.r))
    {
      assert(r.lo() == 0.0f);
    }
  }
}

TEST_HOST_DEVICE_FUNC void test_special_values()
{
  const float inf              = cuda::std::numeric_limits<float>::infinity();
  const float nan              = cuda::std::numeric_limits<float>::quiet_NaN();
  const float maxn             = cuda::std::numeric_limits<float>::max();
  const float minn             = cuda::std::numeric_limits<float>::min();
  const pow_exact_case cases[] = {
    // pow(a, +-0) = 1 for any a, pow(+1, b) = 1 for any b, even NaN
    {nan, 0.0f, 0.0f, 0.0f, 1.0f},
    {nan, 0.0f, -0.0f, 0.0f, 1.0f},
    {1.0f, 0.0f, nan, 0.0f, 1.0f},
    {1.0f, 0.0f, inf, 0.0f, 1.0f},
    {1.0f, 0.0f, -inf, 0.0f, 1.0f},
    // otherwise a NaN operand gives NaN
    {nan, 0.0f, 1.0f, 0.0f, nan},
    {2.0f, 0.0f, nan, 0.0f, nan},
    {-1.0f, 0.0f, nan, 0.0f, nan},
    // pow(+-0, b)
    {0.0f, 0.0f, -1.0f, 0.0f, inf},
    {-0.0f, 0.0f, -1.0f, 0.0f, -inf},
    {-0.0f, 0.0f, -2.0f, 0.0f, inf},
    {-0.0f, 0.0f, -0.5f, 0.0f, inf},
    {-0.0f, 0.0f, -inf, 0.0f, inf},
    {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
    {-0.0f, 0.0f, 1.0f, 0.0f, -0.0f},
    {-0.0f, 0.0f, 2.0f, 0.0f, 0.0f},
    {-0.0f, 0.0f, 0.5f, 0.0f, 0.0f},
    {-0.0f, 0.0f, inf, 0.0f, 0.0f},
    // pow(a, +-inf)
    {-1.0f, 0.0f, inf, 0.0f, 1.0f},
    {-1.0f, 0.0f, -inf, 0.0f, 1.0f},
    {0.5f, 0.0f, -inf, 0.0f, inf},
    {-0.5f, 0.0f, inf, 0.0f, 0.0f},
    {2.0f, 0.0f, -inf, 0.0f, 0.0f},
    {-2.0f, 0.0f, inf, 0.0f, inf},
    {1.0f, -0x1p-30f, inf, 0.0f, 0.0f}, // |a| < 1 only through a_lo
    {1.0f, 0x1p-30f, inf, 0.0f, inf},
    {-1.0f, -0x1p-30f, -inf, 0.0f, 0.0f},
    // pow(+-inf, b), ahead of the NaN for a negative base and a non-integer exponent
    {-inf, 0.0f, -1.0f, 0.0f, -0.0f},
    {-inf, 0.0f, -2.0f, 0.0f, 0.0f},
    {-inf, 0.0f, -0.5f, 0.0f, 0.0f},
    {-inf, 0.0f, 1.0f, 0.0f, -inf},
    {-inf, 0.0f, 2.0f, 0.0f, inf},
    {-inf, 0.0f, 0.5f, 0.0f, inf},
    {inf, 0.0f, -0.5f, 0.0f, 0.0f},
    {inf, 0.0f, 0.5f, 0.0f, inf},
    // a negative base and a non-integer exponent
    {-2.0f, 0.0f, 0.5f, 0.0f, nan},
    {-1.0f, 0.0f, 0x1p24f, 0.5f, nan},
    // exponents that are integers only through b_lo, and odd only through it
    {-1.0f, 0.0f, 0x1p24f, 1.0f, -1.0f},
    {-1.0f, 0.0f, 0x1p24f, 2.0f, 1.0f},
    {-1.0f, 0.0f, 0x1p60f, 0x1p36f, 1.0f},
    {-0.0f, 0.0f, 0x1p24f, 1.0f, -0.0f},
    {-0.0f, 0.0f, -0x1p24f, -1.0f, -inf},
    {-inf, 0.0f, 0x1p24f, 1.0f, -inf},
    {-2.0f, 0.0f, 0x1p24f, 1.0f, -inf},
    {-0.5f, 0.0f, 0x1p24f, 1.0f, -0.0f},
    // b ln|a| far beyond the range
    {maxn, 0.0f, maxn, 0.0f, inf},
    {maxn, 0.0f, -maxn, 0.0f, 0.0f},
    {-maxn, 0.0f, maxn, 0.0f, inf},
    {-maxn, 0.0f, -maxn, 0.0f, 0.0f},
    {minn, 0.0f, -maxn, 0.0f, inf},
    {2.0f, 0.0f, 128.0f, 0.0f, inf},
    {-2.0f, 0.0f, 129.0f, 0.0f, -inf},
    {2.0f, 0.0f, -151.0f, 0.0f, 0.0f},
    {-2.0f, 0.0f, -151.0f, 0.0f, -0.0f},
  };
  check_exact(cases);
}

TEST_HOST_DEVICE_FUNC void test_gradual_underflow()
{
  const float maxn             = cuda::std::numeric_limits<float>::max();
  const float minn             = cuda::std::numeric_limits<float>::min();
  const pow_exact_case cases[] = {
    {2.0f, 0.0f, -140.0f, 0.0f, 0x1p-140f},
    {0.5f, 0.0f, 149.0f, 0.0f, 0x1p-149f},
    {-2.0f, 0.0f, -149.0f, 0.0f, -0x1p-149f},
    {2.0f, 0.0f, -149.5f, 0.0f, 0x1p-149f}, // 2^-149.5 rounds up to the smallest subnormal
    {2.0f, 0.0f, -150.5f, 0.0f, 0.0f}, // and 2^-150.5 down to zero
    {2.0f, 0.0f, -126.5f, 0.0f, 0x1.6a09e8p-127f},
    {minn, 0.0f, 1.0f, 0.0f, minn},
    {0x1p-149f, 0.0f, 1.0f, 0.0f, 0x1p-149f},
    {maxn, 0.0f, -1.0f, 0.0f, 0x1p-128f},
    {-maxn, 0.0f, -1.0f, 0.0f, -0x1p-128f},
  };
  check_exact(cases);
}

int main(int, char**)
{
  test_accuracy();
  test_special_values();
  test_gradual_underflow();

  return 0;
}
