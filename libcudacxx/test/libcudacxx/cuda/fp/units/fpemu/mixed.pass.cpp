// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu mixed arithmetic + mixed-type builtins vs native double.
//
//  Evaluates a complex expression (arithmetic + fma + a conditional) in both
//  native double and fp64emu, and exercises the mixed-type builtins where one
//  argument is fp64emu and the other a plain arithmetic type (double / int).
//  Every emulated result must track its native reference within tolerance.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC void test(double dx, double dy, double dz)
{
  constexpr double c1 = 9.876;
  constexpr int c2    = -6;

  cudax::fp64emu ex = dx;
  cudax::fp64emu ey = dy;
  cudax::fp64emu ez = dz;

  // Complex expression: native double reference vs fp64emu.
  const double ref0 = (dx < dy) ? c2 + (dx * dy + dz) * c2 + cuda::std::fma(dz, dy, dx) / (dz - dx) + c1
                                : c1 + (dx * dz - dy) * c1 + cuda::std::fma(dx, dz, dy) / (dx - dz) + c2;
  const double got0 = (double) ((ex < ey) ? c2 + (ex * ey + ez) * c2 + fma(ez, ey, ex) / (ez - ex) + c1
                                          : c1 + (ex * ez - ey) * c1 + fma(ex, ez, ey) / (ex - ez) + c2);

  // Mixed-type builtins: one fp64emu operand, one plain arithmetic operand.
  const double ref[5] = {dx + 2.5, 2.5 + dx, dx * c2, dx - 1.0, c2 + dy};
  const double got[5] = {
    (double) cudax::__dadd_rn(ex, 2.5),
    (double) cudax::__dadd_rn(2.5, ex),
    (double) cudax::__dmul_rn(ex, c2),
    (double) cudax::__dsub_rn(ex, 1.0),
    (double) cudax::__dadd_rn(c2, ey),
  };

  const double tol = 1e-10;
  assert(cuda::std::fabs(got0 - ref0) <= tol);
  for (int i = 0; i < 5; i++)
  {
    assert(cuda::std::fabs(got[i] - ref[i]) <= tol);
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
