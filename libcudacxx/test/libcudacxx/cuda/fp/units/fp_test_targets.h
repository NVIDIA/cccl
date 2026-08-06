// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  fp_test_targets.h - tiny helpers to announce which target a test ran on.
//
//  The FP unit tests exercise the same run_test() on the host (directly) and, in
//  a CUDA build, on the device (via a kernel launch). These banners make that
//  visible in the log and report the active GPU (name + compute capability) for
//  the device run. Plain functions only -- no test-framework macros -- so the
//  header works identically in the CCCL Catch2/Ninja build and the standalone
//  build.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_FP_UNITS_FP_TEST_TARGETS_H
#define CUDA_FP_UNITS_FP_TEST_TARGETS_H

#include <cuda/std/detail/__config> // _CCCL_CUDA_COMPILATION

#include <cstdio>

inline void fp_ran_on_host()
{
  std::printf("  ran on: host\n");
}

#if _CCCL_CUDA_COMPILATION()
inline void fp_ran_on_device()
{
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
  {
    std::printf("  ran on: device %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  }
  else
  {
    std::printf("  ran on: device (properties unavailable)\n");
  }
}
#endif // _CCCL_CUDA_COMPILATION()

#endif // CUDA_FP_UNITS_FP_TEST_TARGETS_H
