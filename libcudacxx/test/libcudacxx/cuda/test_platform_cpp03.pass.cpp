//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvhpc, nvc++

#include "test_macros.h"
#include <nv/target>

#if !defined(TEST_COMPILER_NVRTC)
#  include <assert.h>
#  include <stdio.h>
#endif

#ifdef __CUDACC__
#  define HD_ANNO __host__ __device__
#else
#  define HD_ANNO
#endif

// Assert macro interferes with preprocessing, wrap it in a function
HD_ANNO inline void check_v(bool result)
{
  assert(result);
}

HD_ANNO void test()
{
#if defined(__CUDA_ARCH__)
  int arch_val = __CUDA_ARCH__;
#else
  int arch_val = 0;
#endif

  unused(arch_val);

  NV_IF_TARGET(NV_IS_HOST, check_v(arch_val == 0);)

  NV_IF_TARGET(NV_IS_DEVICE, check_v(arch_val == __CUDA_ARCH__);)

  NV_IF_ELSE_TARGET(NV_IS_HOST, check_v(arch_val == 0);, check_v(arch_val == __CUDA_ARCH__);)
}

int main(int argc, char** argv)
{
  test();
  return 0;
}
