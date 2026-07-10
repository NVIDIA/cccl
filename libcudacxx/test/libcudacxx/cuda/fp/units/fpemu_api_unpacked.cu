// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu packed + unpacked C++ API (mul/add/mad/dot/poly).
//
//  Exercises the emulated double type through both the packed (fp64emu) and the
//  unpacked (fp64emu_unpacked) C++ API surfaces for a small set of composite
//  operations, and checks each against the native double reference within a tight
//  tolerance. The same _CCCL_HOST_DEVICE run_test() runs on the host and, under
//  CUDA, on the device.
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

#define C0 (1.0)
#define C1 (1.0 / 2.0)
#define C2 (1.0 / 6.0)
#define C3 (1.0 / 24.0)
#define C4 (1.0 / 120.0)
#define C5 (1.0 / 720.0)
#define C6 (1.0 / 5040.0)
#define C7 (1.0 / 40320.0)

// Horner evaluation of the degree-7 polynomial for any value type.
#define POLY(v) (C0 + (v) * (C1 + (v) * (C2 + (v) * (C3 + (v) * (C4 + (v) * (C5 + (v) * (C6 + (v) * C7)))))))

_CCCL_HOST_DEVICE bool run_test(double dx, double dy, double dz, double dw)
{
  const double ref[5] = {
    dx * dy * dz * dw,
    dx + dy + dz + dw,
    dx * dy + dz,
    dx * dy + dz * dw,
    POLY(dx),
  };

  // Packed C++ API.
  fp64emu ex = dx, ey = dy, ez = dz, ew = dw;
  const double packed[5] = {
    (double) (__dmul_rn(ex, ey) * ez * ew),
    (double) (__dadd_rn(ex, ey) + ez + ew),
    (double) mad(ex, ey, ez),
    (double) dot(ex, ez, ey, ew),
    (double) (POLY(ex)),
  };

  // Unpacked C++ API (explicit conversion to disambiguate from the packed type).
  fp64emu_unpacked ux = (fp64emu_unpacked) dx, uy = (fp64emu_unpacked) dy, uz = (fp64emu_unpacked) dz,
                   uw      = (fp64emu_unpacked) dw;
  const double unpacked[5] = {
    (double) (__dmul_rn(ex, ey) * ez * ew),
    (double) (__dadd_rn(ux, uy) + uz + uw),
    (double) mad(ux, uy, uz),
    (double) dot(ux, uz, uy, uw),
    (double) (POLY(ux)),
  };

  const double tol = 1e-10;
  bool ok          = true;
  for (int i = 0; i < 5; i++)
  {
    ok = ok && ::cuda::std::fabs(packed[i] - ref[i]) <= tol;
    ok = ok && ::cuda::std::fabs(unpacked[i] - ref[i]) <= tol;
  }
  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out, double dx, double dy, double dz, double dw)
{
  *out = run_test(dx, dy, dz, dw);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu packed + unpacked C++ API", "[fpemu]")
{
  const double dx = 0.23451432345642;
  const double dy = -2.34561234567899;
  const double dz = 3.45678726352678;
  const double dw = -4.56787263526789;

  fp_ran_on_host();
  REQUIRE(run_test(dx, dy, dz, dw));

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok, dx, dy, dz, dw);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
