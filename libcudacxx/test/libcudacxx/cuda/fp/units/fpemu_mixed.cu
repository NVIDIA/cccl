// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu mixed arithmetic + mixed-type builtins vs native double.
//
//  Evaluates a complex expression (arithmetic + fma + a conditional) in both
//  native double and fp64emu, and exercises the mixed-type builtins where one
//  argument is fp64emu and the other a plain arithmetic type (double / int).
//  Every emulated result must track its native reference within tolerance. The
//  same _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on device.
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

_CCCL_HOST_DEVICE bool run_test(double dx, double dy, double dz)
{
  constexpr double c1 = 9.876;
  constexpr int c2    = -6;

  fp64emu ex = dx;
  fp64emu ey = dy;
  fp64emu ez = dz;

  // Complex expression: native double reference vs fp64emu.
  const double ref0 = (dx < dy) ? c2 + (dx * dy + dz) * c2 + ::cuda::std::fma(dz, dy, dx) / (dz - dx) + c1
                                : c1 + (dx * dz - dy) * c1 + ::cuda::std::fma(dx, dz, dy) / (dx - dz) + c2;
  const double got0 = (double) ((ex < ey) ? c2 + (ex * ey + ez) * c2 + fma(ez, ey, ex) / (ez - ex) + c1
                                          : c1 + (ex * ez - ey) * c1 + fma(ex, ez, ey) / (ex - ez) + c2);

  // Mixed-type builtins: one fp64emu operand, one plain arithmetic operand.
  const double ref[5] = {dx + 2.5, 2.5 + dx, dx * c2, dx - 1.0, c2 + dy};
  const double got[5] = {
    (double) __dadd_rn(ex, 2.5),
    (double) __dadd_rn(2.5, ex),
    (double) __dmul_rn(ex, c2),
    (double) __dsub_rn(ex, 1.0),
    (double) __dadd_rn(c2, ey),
  };

  const double tol = 1e-10;
  bool ok          = ::cuda::std::fabs(got0 - ref0) <= tol;
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

C2H_TEST("fpemu mixed arithmetic + mixed-type builtins", "[fpemu]")
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
