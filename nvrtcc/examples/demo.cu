//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// When compiling with nvrtcc, the __NVRTCC__ macro is defined.
#ifndef __NVRTCC__
#  error "this file must be compiled with nvrtcc"
#endif

// We need to guard includes when compiling device code with nvrtc.
#ifndef __CUDACC_RTC__
#  include <cassert>
#  include <cstdio>
#endif

#define TO_STRING_HELPER(...) #__VA_ARGS__
#define TO_STRING(...)        TO_STRING_HELPER(__VA_ARGS__)

// Ordinary kernels work without any problems.
__global__ void kernel()
{
  printf("[%d, %d]: Hello world from kernel!\n", blockIdx.x, threadIdx.x);
}

// For template kernels the situation is a bit more complicated. We use nvrtcAddNameExpression to make sure these
// kernels are instantiated.
template <class T, int = 3>
__global__ void template_kernel(int)
{
  printf("[%d, %d]: Hello world from template kernel!\n", blockIdx.x, threadIdx.x);

  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    // __CUDA_ARCH_LIST__ behaves the same way as if we were compiling with nvcc.
    printf("\narch list: " TO_STRING(__CUDA_ARCH_LIST__) "\n");
  }
}

// Host functions must not be visible to nvrtc.
#ifndef __CUDACC_RTC__
int main()
{
  // We can now call kernels compiled with nvrtc directly.
  kernel<<<2, 2>>>();
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    return 1;
  }

  printf("\n");

  template_kernel<void><<<3, 1>>>(0);
  if (cudaDeviceSynchronize() != cudaSuccess)
  {
    return 1;
  }
}
#endif
