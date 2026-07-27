// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu accuracy levels.
//
//  Exercises the full set of fpemu accuracy selectors (high / def / low) through
//  the builtin ops (__dadd_rn, __dmul_rn, __dsub_rn, __ddiv_rn, __fma_rn,
//  __dsqrt_rn). The builtins deduce the accuracy level from the argument type, so
//  no explicit template parameters are needed. The result is checked for basic
//  sanity (finite, non-zero, not absurdly large). The same _CCCL_HOST_DEVICE
//  run_test() runs on the host directly and, under CUDA, on the device via a plain
//  kernel that writes its bool result back to managed memory (no
//  __host__ __device__ lambda, so no --extended-lambda dependency).
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

// Computes across all three accuracy levels and verifies the aggregate result is
// finite, non-zero and reasonably bounded. Returns true on success.
_CCCL_HOST_DEVICE bool run_test(double x)
{
  // high accuracy: builtins deduce the accuracy level from the argument type.
  fpemu<double, fpemu_accuracy::high> acc_x = x;
  auto acc_r                                = __dadd_rn(acc_x, acc_x);
  acc_r                                     = __dmul_rn(acc_r, acc_x);
  acc_r                                     = __dsub_rn(acc_r, acc_x);
  acc_r                                     = __ddiv_rn(acc_r, acc_x);
  acc_r                                     = __fma_rn(acc_r, acc_x, acc_x);
  acc_r                                     = __dsqrt_rn(acc_r);

  // default accuracy: fp64emu is fpemu<double, fpemu_accuracy::def> (== high).
  fp64emu def_x = x;
  auto def_r    = __dadd_rn(def_x, def_x);
  def_r         = __dmul_rn(def_r, def_x);
  def_r         = __dsub_rn(def_r, def_x);
  def_r         = __ddiv_rn(def_r, def_x);
  def_r         = __fma_rn(def_r, def_x, def_x);
  def_r         = __dsqrt_rn(def_r);

  // low accuracy: builtins deduce the accuracy level from the argument type.
  fpemu<double, fpemu_accuracy::low> fast_x = x;
  auto fast_r                               = __dadd_rn(fast_x, fast_x);
  fast_r                                    = __dmul_rn(fast_r, fast_x);
  fast_r                                    = __dsub_rn(fast_r, fast_x);
  fast_r                                    = __ddiv_rn(fast_r, fast_x);

  const double r = (double) acc_r + (double) def_r + (double) fast_r;

  // Sanity: not NaN, not zero, and not absurdly large.
  return (r == r) && (r != 0.0) && (::cuda::std::fabs(r) < 1e20);
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out, double x)
{
  *out = run_test(x);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpemu accuracy levels", "[fpemu]")
{
  const double x = 1.2345;

  // Host run.
  fp_ran_on_host();
  REQUIRE(run_test(x));

#if _CCCL_CUDA_COMPILATION()
  // Device run: same run_test() in a kernel, result read back via managed memory.
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok, x);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
