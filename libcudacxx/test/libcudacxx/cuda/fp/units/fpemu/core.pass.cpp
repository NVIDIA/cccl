// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu core builtins on raw __fpbits64 vs native double.
//
//  Drives the low-level emulation builtins (__fp64emu_from_double / _to_double
//  and the accuracy-tagged __fp64emu_*_dmul_rn / _dadd_rn / _ddiv_rn / _dsub_rn /
//  _fma_rn cores) and checks each against the native double reference within a
//  tight tolerance.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Runs mul/add/div/sub/fma through the raw __fpbits64 emulation cores and checks
// each against the native double reference. Returns true on success.
TEST_HOST_DEVICE_FUNC void test(double dx, double dy, double dz)
{
  cudax::__fpbits64 ex = cudax::__fp64emu_from_double(dx);
  cudax::__fpbits64 ey = cudax::__fp64emu_from_double(dy);
  cudax::__fpbits64 ez = cudax::__fp64emu_from_double(dz);

  const double ref[5] = {dx * dy, dx + dy, dx / dy, dx - dy, dx * dy + dz};

  const double got[5] = {
    cudax::__fp64emu_to_double(cudax::__fp64emu_mid_dmul_rn(ex, ey)),
    cudax::__fp64emu_to_double(cudax::__fp64emu_high_dadd_rn(ex, ey)),
    cudax::__fp64emu_to_double(cudax::__fp64emu_mid_ddiv_rn(ex, ey)),
    cudax::__fp64emu_to_double(cudax::__fp64emu_high_dsub_rn(ex, ey)),
    cudax::__fp64emu_to_double(cudax::__fp64emu_mid_fma_rn(ex, ey, ez)),
  };

  const double tol = 1e-10;
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
