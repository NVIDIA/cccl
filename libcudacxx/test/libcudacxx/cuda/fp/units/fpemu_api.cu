// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu C++ API (operators + builtins) vs native double.
//
//  Exercises the emulated double type fp64emu for the basic ops (mul, add, div,
//  sub) and fma, using both the operator interface (ex * ey, ...) and the
//  accuracy-selecting builtins (__dmul_rn, __dadd_rn, ...). Each emulated result
//  must track the native double reference to a tight tolerance. The same
//  _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the device.
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

// Runs each op through the fp64emu operators and the builtins, and verifies both
// track the native double reference within tolerance. Returns true on success.
_CCCL_HOST_DEVICE bool run_test(double dx, double dy, double dz)
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
  bool ok          = true;
  for (int i = 0; i < 5; i++)
  {
    ok = ok && ::cuda::std::fabs(cpp[i] - ref[i]) <= tol;
    ok = ok && ::cuda::std::fabs(bi[i] - ref[i]) <= tol;
  }
  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out, double dx, double dy, double dz)
{
  *out = run_test(dx, dy, dz);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu C++ API vs native double", "[fpemu]")
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
