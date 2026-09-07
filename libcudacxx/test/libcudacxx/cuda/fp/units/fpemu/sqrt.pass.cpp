// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu square root (correctly rounded, bit-exact).
//
//  Validates that the fpemu square root reproduces, bit-for-bit, correctly-rounded
//  IEEE-754 binary64 sqrt for all four rounding modes (rn, rz, ru, rd) across the C
//  builtins (__fp64emu_dsqrt_*), the packed sqrt (rn) and the unpacked sqrt (rn).
//  The reference is the CUDA __dsqrt_* intrinsics on the device and fenv-directed
//  sqrt on the host, so the same check runs on the host and, under CUDA, on the
//  device. NaN results are matched by class.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/random>

#if _CCCL_HOST_COMPILATION()
#  include <cfenv>
#endif

#include <nv/target>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

enum
{
  M_RN = 0,
  M_RZ,
  M_RU,
  M_RD,
  M_COUNT
};

TEST_HOST_DEVICE_FUNC double from_bits(uint64_t b)
{
  return cuda::std::bit_cast<double>(b);
}
TEST_HOST_DEVICE_FUNC uint64_t d_bits(double d)
{
  return cuda::std::bit_cast<uint64_t>(d);
}

TEST_HOST_DEVICE_FUNC bool is_nan_bits(uint64_t b)
{
  return ((b & UINT64_C(0x7FF0000000000000)) == UINT64_C(0x7FF0000000000000)) && (b & UINT64_C(0x000FFFFFFFFFFFFF));
}
// NaN payloads are platform-defined: treat any two NaNs as a match.
TEST_HOST_DEVICE_FUNC bool match(uint64_t got, uint64_t ref)
{
  return (got == ref) || (is_nan_bits(got) && is_nan_bits(ref));
}

// Reference: CUDA __dsqrt_* intrinsics on device, fenv-directed sqrt on host.
TEST_HOST_DEVICE_FUNC uint64_t ref_one(double a, int mode){NV_IF_ELSE_TARGET(
  NV_IS_DEVICE,
  ({
    double r;
    switch (mode)
    {
      case M_RZ:
        r = ::__dsqrt_rz(a);
        break;
      case M_RU:
        r = ::__dsqrt_ru(a);
        break;
      case M_RD:
        r = ::__dsqrt_rd(a);
        break;
      default:
        r = ::__dsqrt_rn(a);
        break; // M_RN
    }
    return d_bits(r);
  }),
  ({
    int old = fegetround();
    int fe;
    switch (mode)
    {
      case M_RZ:
        fe = FE_TOWARDZERO;
        break;
      case M_RU:
        fe = FE_UPWARD;
        break;
      case M_RD:
        fe = FE_DOWNWARD;
        break;
      default:
        fe = FE_TONEAREST;
        break; // M_RN
    }
    fesetround(fe);
    // volatile forces the sqrtsd to execute (in memory) between the fesetround
    // calls and prevents compile-time constant folding under the wrong mode.
    volatile double va = a;
    volatile double r  = ::sqrt(va);
    double rr          = r;
    fesetround(old);
    return d_bits(rr);
  }))}

// Compare every sqrt surface for one value against the reference on the same
// target.
TEST_HOST_DEVICE_FUNC void check_value(double x)
{
  cudax::__fpbits64 a = cudax::__fp64emu_from_double(x);

  assert(match((uint64_t) cudax::__fp64emu_dsqrt_rn(a), ref_one(x, M_RN)));
  assert(match((uint64_t) cudax::__fp64emu_dsqrt_rz(a), ref_one(x, M_RZ)));
  assert(match((uint64_t) cudax::__fp64emu_dsqrt_ru(a), ref_one(x, M_RU)));
  assert(match((uint64_t) cudax::__fp64emu_dsqrt_rd(a), ref_one(x, M_RD)));

  cudax::fp64emu p = x;
  assert(match(d_bits((double) sqrt(p)), ref_one(x, M_RN)));

  cudax::fp64emu_unpacked u = (cudax::fp64emu_unpacked) x;
  assert(match(d_bits((double) sqrt(u)), ref_one(x, M_RN)));
}

// The four classes the old randomized sweep drew from. The two uniform ranges are
// non-negative, where the result is a real number; negative inputs still arrive
// through the special values and the arbitrary bit patterns.
TEST_HOST_DEVICE_FUNC double draw(cuda::std::minstd_rand& rng, const double* specials, int n)
{
  cuda::std::uniform_int_distribution<int> which(0, 3);
  cuda::std::uniform_int_distribution<int> pick(0, n - 1);
  cuda::std::uniform_int_distribution<uint64_t> bits;
  cuda::std::uniform_real_distribution<double> small(0.0, 16.0);
  cuda::std::uniform_real_distribution<double> wide(0.0, 1.0e150);

  switch (which(rng))
  {
    case 0:
      return specials[pick(rng)];
    case 1:
      return small(rng);
    case 2:
      return wide(rng);
    default:
      return cuda::std::bit_cast<double>(bits(rng));
  }
}

// Takes the square root of the representative special values and checks each
// surface against the correctly-rounded reference, then repeats the check over a
// deterministic pseudo-random sweep.
TEST_FUNC void test()
{
  const double specials[] = {
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    3.0,
    4.0,
    0.5,
    0.25,
    100.0,
    -100.0,
    3.14159265358979,
    1e-300,
    1e300,
    -1e-300,
    from_bits(0x0000000000000001ULL), // min subnormal
    from_bits(0x000FFFFFFFFFFFFFULL), // max subnormal
    from_bits(0x0010000000000000ULL), // min normal
    from_bits(0x7FEFFFFFFFFFFFFFULL), // max normal
    from_bits(0x7FF0000000000000ULL), // +inf
    from_bits(0xFFF0000000000000ULL), // -inf
    from_bits(0x7FF8000000000000ULL), // +qNaN
    from_bits(0xFFF8000000000000ULL), // -qNaN
    from_bits(0x7FF0000000000001ULL), // +sNaN
  };
  const int n = (int) (sizeof(specials) / sizeof(specials[0]));

  for (int i = 0; i < n; i++)
  {
    check_value(specials[i]);
  }

  cuda::std::minstd_rand rng(0x5417u);
  for (int i = 0; i < 512; i++)
  {
    check_value(draw(rng, specials, n));
  }
}

int main(int, char**)
{
  test();

  return 0;
}
