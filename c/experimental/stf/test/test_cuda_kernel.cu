//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <c2h/catch2_test_helper.h>
#include <cccl/c/experimental/stf/stf.h>

#if 0
__global__ void axpy(int cnt, double a, const double *x, double *y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

//  for (int i = tid; i < cnt; i += nthreads)
//  {
//    y[i] += a * x[i];
//  }
}
#endif

extern "C" __global__ void axpy(int, double, const double*, double*)
{
  printf("hello.\n");
}

C2H_TEST("axpy with stf cuda_kernel", "[cuda_kernel]")
{
  size_t N = 1000000;

  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  stf_logical_data_handle lX, lY;

  float *X, *Y;
  X = (float*) malloc(N * sizeof(float));
  Y = (float*) malloc(N * sizeof(float));

  stf_logical_data(ctx, &lX, X, N * sizeof(float));
  stf_logical_data(ctx, &lY, Y, N * sizeof(float));

  stf_logical_data_set_symbol(lX, "X");
  stf_logical_data_set_symbol(lY, "Y");

  stf_cuda_kernel_handle k;
  stf_cuda_kernel_create(ctx, &k);
  stf_cuda_kernel_set_symbol(k, "axpy");
  stf_cuda_kernel_add_dep(k, lX, STF_READ);
  stf_cuda_kernel_add_dep(k, lY, STF_RW);
  stf_cuda_kernel_start(k);
  void* dummy         = nullptr;
  const void* args[4] = {&N, &alpha, &dummy, &dummy};
  stf_cuda_kernel_add_desc(k, (void*) axpy, 2, 4, 0, 4, args);
  stf_cuda_kernel_end(k);
  stf_cuda_kernel_destroy(k);

  stf_logical_data_destroy(lX);
  stf_logical_data_destroy(lY);

  stf_ctx_finalize(ctx);

  free(X);
  free(Y);
}
