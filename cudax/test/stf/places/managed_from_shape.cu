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
 * @brief Make sure we can automatically allocate and use data in managed memory based on their shape
 *
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <iostream>

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

__host__ __device__ double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

int main()
{
  // Verify whether this device can access memory concurrently from CPU and GPU.
  int dev;
  cuda_safe_call(cudaGetDevice(&dev));
  assert(dev >= 0);
  cudaDeviceProp prop;
  cuda_safe_call(cudaGetDeviceProperties(&prop, dev));
  if (!prop.concurrentManagedAccess)
  {
    fprintf(stderr, "Concurrent CPU/GPU access not supported, skipping test.\n");
    return 0;
  }

  stream_ctx ctx;
  const size_t N = 16;
  double Y[N];

  for (size_t i = 0; i < N; i++)
  {
    Y[i] = Y0(i);
  }

  double alpha = 3.14;

  auto lX = ctx.logical_data(shape_of<slice<double>>(N));
  auto lY = ctx.logical_data(Y);

  // Make sure X is created automatically in managed memory
  ctx.parallel_for(lX.shape(), lX.write(data_place::managed))->*[] _CCCL_DEVICE(size_t i, auto X) {
    X(i) = X0(i);
  };

  /* Compute Y = Y + alpha X, but leave X in managed memory */
  ctx.task(lX.read(data_place::managed), lY.rw())->*[&](cudaStream_t s, auto dX, auto dY) {
    axpy<<<16, 128, 0, s>>>(alpha, dX, dY);
  };

  ctx.host_launch(lX.read(data_place::managed), lY.read())->*[=](auto X, auto Y) {
    for (size_t i = 0; i < N; i++)
    {
      EXPECT(fabs(Y(i) - (Y0(i) + alpha * X0(i))) < 0.0001);
      EXPECT(fabs(X(i) - X0(i)) < 0.0001);
    }
  };

  ctx.finalize();
}
