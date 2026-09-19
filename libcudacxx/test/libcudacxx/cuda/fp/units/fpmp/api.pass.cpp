// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: float-float arithmetic API (fp32mp2).
//
//  Compares native double-precision arithmetic against the float-float type
//  fp32mp2 (== fpmp2<float, fpmp2_accuracy::def>) for the basic ops (mul, add,
//  div, sub) and fma. fp32mp2 carries ~46 effective mantissa bits, so its results
//  must track the double reference to a tight relative tolerance.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point type.
using ffloat = cudax::fp32mp2;

// Relative-error check against a double reference. fp32mp2 keeps ~46 mantissa
// bits (~1.4e-14 relative), so 1e-10 is a safe, still-meaningful bound.
TEST_HOST_DEVICE_FUNC bool close(double got, double ref)
{
  const double scale = ::cuda::std::fabs(ref) > 1.0 ? ::cuda::std::fabs(ref) : 1.0;
  return ::cuda::std::fabs(got - ref) <= 1e-10 * scale;
}

// Runs each op in float-float precision and verifies it matches the double
// reference within tolerance. Returns true on success.
TEST_HOST_DEVICE_FUNC void run_test(double dx, double dy, double dz)
{
  // double -> fp32mp2 is a narrowing conversion, so construct explicitly.
  ffloat ex = ffloat(dx);
  ffloat ey = ffloat(dy);
  ffloat ez = ffloat(dz);

  assert(close((double) (ex * ey), dx * dy));
  assert(close((double) (ex + ey), dx + dy));
  assert(close((double) (ex / ey), dx / dy));
  assert(close((double) (ex - ey), dx - dy));
  assert(close((double) fma(ex, ey, ez), ::cuda::std::fma(dx, dy, dz)));
}

TEST_HOST_DEVICE_FUNC void test()
{
  // High-precision constants (as in the original example).
  const double dx = 1.123456782345678936;
  const double dy = 2.234567891234567856;
  const double dz = 3.345678901234567892;

  run_test(dx, dy, dz);
}

int main(int, char**)
{
  test();

  return 0;
}
