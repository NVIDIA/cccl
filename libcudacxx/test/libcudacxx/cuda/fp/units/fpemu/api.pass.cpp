// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu C++ API (operators + builtins) vs native double.
//
//  Exercises the emulated double type fp64emu for the basic ops (mul, add, div,
//  sub) and fma, using both the operator interface (ex * ey, ...) and the
//  accuracy-selecting builtins (__dmul_rn, __dadd_rn, ...). Each emulated result
//  must track the native double reference to a tight tolerance.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Runs each op through the fp64emu operators and the builtins, and verifies both
// track the native double reference within tolerance. Returns true on success.
TEST_HOST_DEVICE_FUNC void test(double dx, double dy, double dz)
{
  fp64emu ex = dx;
  fp64emu ey = dy;
  fp64emu ez = dz;

  const double ref[5] = {dx * dy, dx + dy, dx / dy, dx - dy, dx * dy + dz};

  const double cpp[5] = {
    (double) (ex * ey),
    (double) (ex + ey),
    (double) (ex / ey),
    (double) (ex - ey),
    (double) fma(ex, ey, ez),
  };

  const double bi[5] = {
    (double) __dmul_rn(ex, ey),
    (double) __dadd_rn(ex, ey),
    (double) __ddiv_rn(ex, ey),
    (double) __dsub_rn(ex, ey),
    (double) __fma_rn(ex, ey, ez),
  };

  const double tol = 1e-10;
  for (int i = 0; i < 5; i++)
  {
    assert(cuda::std::fabs(cpp[i] - ref[i]) <= tol);
    assert(cuda::std::fabs(bi[i] - ref[i]) <= tol);
  }
}

int main(int, char**)
{
  const double dx = 1.2345;
  const double dy = 2.3456;
  const double dz = 3.4567;
  test(dx, dy, dz);

  return 0;
}
