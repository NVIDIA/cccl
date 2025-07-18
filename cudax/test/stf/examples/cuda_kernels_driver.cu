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

int main()
{
  context ctx;
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

  CUfunction axpy_fun;
  cuda_safe_call(cudaGetFuncBySymbol(&axpy_fun, (void*) axpy));

  // TODO ifdef
  CUkernel axpy_kernel;
  cuda_safe_call(cudaGetKernel(&axpy_kernel, (void*) axpy));

  // runtime global kernel
  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto dX, auto dY) {
    // axpy<<<16, 128, 0, ...>>>(alpha, dX, dY)
    return cuda_kernel_desc{axpy, 16, 128, 0, alpha, dX, dY};
  };
  num_axpy++;

  // CUfunction driver API
  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto dX, auto dY) {
    return cuda_kernel_desc{axpy_fun, 16, 128, 0, alpha, dX, dY};
  };
  num_axpy++;

#if CUDA_VERSION >= 12000
  // CUkernel driver API
  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto dX, auto dY) {
    return cuda_kernel_desc{axpy_kernel, 16, 128, 0, alpha, dX, dY};
  };
  num_axpy++;
#endif

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + num_axpy * alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }
}
