// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: exp() implemented on fp64emu vs std::exp.
//
//  A generic exp_impl<T>() (polynomial approximation + range reduction) is
//  instantiated for the emulated double fp64emu and its result is compared against
//  the native std::exp reference with a relative-error bound. The same generic
//  code path proves fp64emu can back an application-level transcendental.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

constexpr double epsilon = 1e-4;

// exp() via range reduction + polynomial, generic over double and fp64emu.
template <typename T>
TEST_HOST_DEVICE_FUNC T exp_impl(T x)
{
  constexpr double ln2_hi  = 0x1.62e42fefa39efp-1; // high part of ln(2)
  constexpr double ln2_lo  = 0x1.abc9e3b39803fp-34; // low part for extra precision
  constexpr double inv_ln2 = 0x1.71547652b82fep+0; // 1 / ln(2)

  if (x != x)
  {
    return x; // NaN
  }
  if (x > 709.782712893384)
  {
    return T(1.0) / 0.0; // overflow
  }
  if (x < -745.1332191019411)
  {
    return T(0.0); // underflow
  }

  // Range reduction: x = k * ln2 + r,  |r| <= ln2/2.
  int k = (int) (x * inv_ln2 + (x >= 0 ? 0.5 : -0.5));

  T r;
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    ({
      r = __fma_rn(-k, ln2_hi, x);
      r = __fma_rn(-k, ln2_lo, r);
    }),
    ({
      r = fma(-k, ln2_hi, x);
      r = fma(-k, ln2_lo, r);
    }))

  // Polynomial approximation of exp(r), r in [-ln2/2, ln2/2].
  T poly =
    0x1p+0
    + r
        * (0x1p+0
           + r
               * (0x1p-1
                  + r
                      * (0x1.5555555555555p-3
                         + r * (0x1.999999999999ap-5 + r * (0x1.6c16c16c16c17p-7 + r * (0x1.a01a01a01a01ap-9))))));

  // Reconstruct exp(x) = 2^k * exp(r). Bias = 1023 for double.
  int exponent = k + 1023;
  if (exponent <= 0) // subnormal
  {
    if (exponent < -52)
    {
      return T(0.0);
    }
    uint64_t uexp = (uint64_t) (exponent + 52) << 52;
    T dexp        = cuda::std::bit_cast<T>(uexp);
    return poly * dexp * 0x1.0p-52;
  }

  if (exponent >= 2047)
  {
    return T(1.0) / 0.0;
  }

  uint64_t uexp = (uint64_t) exponent << 52;
  T dexp        = cuda::std::bit_cast<T>(uexp);
  return poly * dexp;
}

// Evaluate exp_impl<fp64emu> across a fixed set of inputs and verify each stays
// within the relative-error bound of std::exp.
TEST_HOST_DEVICE_FUNC void test()
{
  const double tv[10] = {0.0, 0.00001, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0, 700.0, -700.0};

  for (const double x : tv)
  {
    const double ref = cuda::std::exp(x);
    const double got = (double) exp_impl<cudax::fp64emu>(x);
    const double rel = (ref != 0.0) ? cuda::std::fabs(got - ref) / ref : 0.0;
    assert(rel < epsilon);
  }
}

int main(int, char**)
{
  test();

  return 0;
}
