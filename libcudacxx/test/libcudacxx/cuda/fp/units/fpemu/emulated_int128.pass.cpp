// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu cores with the emulated 128-bit integer
//  (_CCCL_FPEMU_FORCE_EMULATED_INT128), the path NVRTC takes without
//  -device-int128.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#define _CCCL_FPEMU_FORCE_EMULATED_INT128

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

TEST_HOST_DEVICE_FUNC void test(double dx, double dy, double dz)
{
  cuda::__fpbits64 ex = cuda::__fp64emu_from_double(dx);
  cuda::__fpbits64 ey = cuda::__fp64emu_from_double(dy);
  cuda::__fpbits64 ez = cuda::__fp64emu_from_double(dz);

  const double ref[5] = {dx * dy, dx + dy, dx / dy, dx - dy, dx * dy + dz};

  const double got[5] = {
    cuda::__fp64emu_to_double(cuda::__fp64emu_mid_dmul_rn(ex, ey)),
    cuda::__fp64emu_to_double(cuda::__fp64emu_high_dadd_rn(ex, ey)),
    cuda::__fp64emu_to_double(cuda::__fp64emu_mid_ddiv_rn(ex, ey)),
    cuda::__fp64emu_to_double(cuda::__fp64emu_high_dsub_rn(ex, ey)),
    cuda::__fp64emu_to_double(cuda::__fp64emu_mid_fma_rn(ex, ey, ez)),
  };

  const double tol = 1e-10;
  for (int i = 0; i < 5; i++)
  {
    assert(cuda::std::fabs(got[i] - ref[i]) <= tol);
  }
}

int main(int, char**)
{
  test(1.2345, 2.3456, 3.4567);
  return 0;
}
