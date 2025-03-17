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
 * @brief An AXPY kernel implemented with a task of the CUDA graph backend and
 *        a host callback
 *
 * The host_launch mechanism is also illustrated
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
  graph_ctx ctx;
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  /* Compute Y = Y + alpha X */
  ctx.task(lX.read(), lY.rw())->*[&](cudaStream_t s, auto dX, auto dY) {
    axpy<<<16, 128, 0, s>>>(alpha, dX, dY);
  };

  /* Asynchronously check the result on the host */
  ctx.host_launch(lX.read(), lY.read())->*[&](auto hX, auto hY) {
    for (size_t ind = 0; ind < hX.extent(0); ind++)
    {
      // Y should be Y0 + alpha X0
      EXPECT(fabs(hY(ind) - (Y0(ind) + alpha * X0(ind))) < 0.0001);

      // X should be X0
      EXPECT(fabs(hX(ind) - X0(ind)) < 0.0001);
    }
  };

  ctx.finalize();
}
