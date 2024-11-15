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
 * @brief An AXPY kernel implemented using the parallel_for construct
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

class sum {
public:
};

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

  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  auto lsum = ctx.logical_data(shape_of<slice<double>>(1));
  auto lsum2 = ctx.logical_data(shape_of<slice<double>>(1));

  /* Compute Y = Y + alpha X */
  //ctx.parallel_for(lY.shape(), lX.read(), lY.rw(), lsum.write(), lsum2.template reduce<sum>())->*[alpha] __device__(size_t i, auto dX, auto dY, auto sum, auto sum2) {
  ctx.parallel_for(lY.shape(), lX.read(), lY.rw(), lsum.write(), lsum2.reduce(sum{}))->*[alpha] __device__(size_t i, auto dX, auto dY, auto sum, auto sum2) {
    dY(i) += alpha * dX(i);
    // sum(0) += dY(i);
    atomicAdd(sum.data_handle(), dY(i));
  };

  ctx.host_launch(lsum.read())->*[](auto sum) {
      fprintf(stderr, "SUM ... %lf\n", sum(0));
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }
}
