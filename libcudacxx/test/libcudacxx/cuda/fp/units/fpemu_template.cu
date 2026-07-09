// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu accuracy template parameter selection.
//
//  Demonstrates selecting the emulation accuracy at compile time via
//  fpemu<double, m>. The expression ((x + x) * x - x) / (x + c) is evaluated with
//  high / def / low accuracy and each result is checked against the native double
//  reference with an accuracy-appropriate tolerance (tight for high, relaxed for
//  low). The same _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA,
//  on the device.
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

// Evaluate ((x + x) * x - x) / (x + c) with a chosen accuracy, using the builtins
// (which deduce the accuracy from their fpemu<double, m> argument types).
template <fpemu_accuracy m>
_CCCL_HOST_DEVICE double run_mode(double x0)
{
  fpemu<double, m> x = x0;
  fpemu<double, m> c = 0.001;
  return (double) __ddiv_rn(__dsub_rn(__dmul_rn(__dadd_rn(x, x), x), x), __dadd_rn(x, c));
}

_CCCL_HOST_DEVICE bool run_test(double x0)
{
  const double c   = 0.001;
  const double ref = ((x0 + x0) * x0 - x0) / (x0 + c);

  const double hi = run_mode<fpemu_accuracy::high>(x0);
  const double de = run_mode<fpemu_accuracy::def>(x0);
  const double lo = run_mode<fpemu_accuracy::low>(x0);

  return ::cuda::std::fabs(hi - ref) <= 1e-12 // high accuracy: tight
      && ::cuda::std::fabs(de - ref) <= 1e-10 // def  accuracy
      && ::cuda::std::fabs(lo - ref) <= 1e-4; // low  accuracy: relaxed
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out, double x0)
{
  *out = run_test(x0);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu accuracy template selection", "[fpemu]")
{
  const double x0 = -0x1.57f1782782a8ap-1;

  fp_ran_on_host();
  REQUIRE(run_test(x0));

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok, x0);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
