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

__global__ void axpy(int cnt, double a, const double* x, double* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < cnt; i += nthreads)
  {
    y[i] += a * x[i];
  }
}

double X0(int i)
{
  return sin(static_cast<double>(i));
}

double Y0(int i)
{
  return cos((double) i);
}

C2H_TEST("axpy with stf cuda_kernel", "[cuda_kernel]")
{
  size_t N = 1000000;

  stf_ctx_handle ctx;
  stf_ctx_create(&ctx);

  stf_logical_data_handle lX, lY;

  double *X, *Y;
  X = (double*) malloc(N * sizeof(double));
  Y = (double*) malloc(N * sizeof(double));

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  const double alpha = 3.14;

  stf_logical_data(ctx, &lX, X, N * sizeof(double));
  stf_logical_data(ctx, &lY, Y, N * sizeof(double));

  stf_logical_data_set_symbol(lX, "X");
  stf_logical_data_set_symbol(lY, "Y");

  stf_cuda_kernel_handle k;
  stf_cuda_kernel_create(ctx, &k);
  stf_cuda_kernel_set_symbol(k, "axpy");
  stf_cuda_kernel_add_dep(k, lX, STF_READ);
  stf_cuda_kernel_add_dep(k, lY, STF_RW);
  stf_cuda_kernel_start(k);
  double* dX          = (double*) stf_cuda_kernel_get_arg(k, 0);
  double* dY          = (double*) stf_cuda_kernel_get_arg(k, 1);
  const void* args[4] = {&N, &alpha, &dX, &dY};
  cudaError_t err     = stf_cuda_kernel_add_desc(k, (void*) axpy, 2, 4, 0, 4, args);
  REQUIRE(err == cudaSuccess);
  stf_cuda_kernel_end(k);
  stf_cuda_kernel_destroy(k);

  stf_logical_data_destroy(lX);
  stf_logical_data_destroy(lY);

  stf_ctx_finalize(ctx);

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }

  free(X);
  free(Y);
}
