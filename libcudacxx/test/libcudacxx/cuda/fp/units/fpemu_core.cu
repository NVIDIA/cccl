// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu core builtins on raw __fpbits64 vs native double.
//
//  Drives the low-level emulation builtins (__fp64emu_from_double / _to_double
//  and the accuracy-tagged __fp64emu_*_dmul_rn / _dadd_rn / _ddiv_rn / _dsub_rn /
//  _fma_rn cores) and checks each against the native double reference within a
//  tight tolerance. The same _CCCL_HOST_DEVICE run_test() runs on the host and,
//  under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Runs mul/add/div/sub/fma through the raw __fpbits64 emulation cores and checks
// each against the native double reference. Returns true on success.
_CCCL_HOST_DEVICE bool run_test(double dx, double dy, double dz)
{
  __fpbits64 ex = __fp64emu_from_double(dx);
  __fpbits64 ey = __fp64emu_from_double(dy);
  __fpbits64 ez = __fp64emu_from_double(dz);

  const double ref[5] = {dx * dy, dx + dy, dx / dy, dx - dy, dx * dy + dz};

  const double got[5] = {
    __fp64emu_to_double(__fp64emu_mid_dmul_rn(ex, ey)),
    __fp64emu_to_double(__fp64emu_high_dadd_rn(ex, ey)),
    __fp64emu_to_double(__fp64emu_mid_ddiv_rn(ex, ey)),
    __fp64emu_to_double(__fp64emu_high_dsub_rn(ex, ey)),
    __fp64emu_to_double(__fp64emu_mid_fma_rn(ex, ey, ez)),
  };

  const double tol = 1e-10;
  bool ok          = true;
  for (int i = 0; i < 5; i++)
  {
    ok = ok && ::cuda::std::fabs(got[i] - ref[i]) <= tol;
  }
  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out, double dx, double dy, double dz)
{
  *out = run_test(dx, dy, dz);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu core builtins vs native double", "[fpemu]")
{
  const double dx = 1.2345;
  const double dy = 2.3456;
  const double dz = 3.4567;

  fp_ran_on_host();
  REQUIRE(run_test(dx, dy, dz));

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok, dx, dy, dz);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
