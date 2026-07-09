// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: bit_cast on fp64emu_unpacked values.
//
//  Verifies that bit_cast between the unpacked emulated double and its IEEE-754
//  bit representation round-trips values exactly, produces the expected result of
//  a simple arithmetic expression, and yields identical bits for a plain
//  conversion across all accuracy levels. The same _CCCL_HOST_DEVICE run_test()
//  runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpemu>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok = true;

  // Round-trip: double -> unpacked -> bit_cast<double> must preserve the value.
  const double test_vals[5] = {1.5, -2.0, 0.0, 42.0, 3.14159265358979323846};
  for (int i = 0; i < 5; i++)
  {
    fpemu_unpacked<double, fpemu_accuracy::def> x(test_vals[i]);
    ok = ok && (bit_cast<double>(x) == test_vals[i]);
  }

  // Arithmetic result: 2 * 3 + 1 == 7.
  fpemu_unpacked<double, fpemu_accuracy::def> a(2.0), b(3.0), c(1.0);
  ok = ok && (::cuda::std::fabs(bit_cast<double>(a * b + c) - 7.0) <= 1e-10);

  // A plain conversion produces identical bits across all accuracy levels.
  const double pi        = 3.14159265358979323846;
  const uint64_t bits_def  = bit_cast<uint64_t>(fpemu_unpacked<double, fpemu_accuracy::def>(pi));
  const uint64_t bits_high = bit_cast<uint64_t>(fpemu_unpacked<double, fpemu_accuracy::high>(pi));
  const uint64_t bits_low  = bit_cast<uint64_t>(fpemu_unpacked<double, fpemu_accuracy::low>(pi));
  ok = ok && (bits_def == bits_high) && (bits_def == bits_low);

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu_unpacked bit_cast", "[fpemu]")
{
  fp_ran_on_host();
  REQUIRE(run_test());

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
