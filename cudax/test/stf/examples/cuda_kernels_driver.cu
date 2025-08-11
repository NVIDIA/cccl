//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Test that the cuda_kernel construct works with global kernels, CUfunction and CUkernel entries.
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < x.size(); i += nthreads)
  {
    y(i) += a * x(i);
  }
}

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

void test(bool is_graph)
{
  context ctx;
  if (is_graph)
  {
    ctx = graph_ctx();
  }

  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  // Number of times we have applied the axpy kernel
  int num_axpy = 0;

  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  // runtime global kernel
  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto dX, auto dY) {
    // axpy<<<16, 128, 0, ...>>>(alpha, dX, dY)
    return cuda_kernel_desc{axpy, 16, 128, 0, alpha, dX, dY};
  };
  num_axpy++;

  // CUfunction driver API
  CUfunction axpy_fun;
  cuda_safe_call(cudaGetFuncBySymbol(&axpy_fun, (void*) axpy));

  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto dX, auto dY) {
    return cuda_kernel_desc{axpy_fun, 16, 128, 0, alpha, dX, dY};
  };
  num_axpy++;

#if _CCCL_CTK_AT_LEAST(12, 1)
  // CUkernel driver API
  CUkernel axpy_kernel;
  cuda_safe_call(cudaGetKernel(&axpy_kernel, (void*) axpy));

  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto dX, auto dY) {
    return cuda_kernel_desc{axpy_kernel, 16, 128, 0, alpha, dX, dY};
  };
  num_axpy++;
#endif

  /* Some extra sanity checks, we put this in a dummy task to get access to dX and dY values */
  ctx.task(lX.read(), lY.rw())->*[&](auto, auto dX, auto dY) {
    int nregs = cuda_kernel_desc{axpy, 16, 128, 0, alpha, dX, dY}.get_num_registers();

    int nregs_fun = cuda_kernel_desc{axpy_fun, 16, 128, 0, alpha, dX, dY}.get_num_registers();
    _CCCL_ASSERT(nregs == nregs_fun, "invalid value");

#if _CCCL_CTK_AT_LEAST(12, 1)
    int nregs_kernel = cuda_kernel_desc{axpy_kernel, 16, 128, 0, alpha, dX, dY}.get_num_registers();
    _CCCL_ASSERT(nregs == nregs_kernel, "invalid value");
#endif
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    _CCCL_ASSERT(fabs(Y[i] - (Y0(i) + num_axpy * alpha * X0(i))) < 0.0001, "Invalid result");
    _CCCL_ASSERT(fabs(X[i] - X0(i)) < 0.0001, "Invalid result");
  }
}

int main()
{
  // stream context
  test(false);
  // graph context
  test(true);
}
