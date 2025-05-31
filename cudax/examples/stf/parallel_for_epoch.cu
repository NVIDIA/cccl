//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief An AXPY kernel implemented with a task of the CUDA stream backend
 * where the task accesses managed memory from the device. This tests
 * explicitly created managed memory, and passes it to a logical data.
 */

#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

int main()
{
  context ctx;
  const size_t N = 16384;

  double *X, *Y, *Z;
  cuda_safe_call(cudaMallocManaged(&X, N * sizeof(double)));
  cuda_safe_call(cudaMallocManaged(&Y, N * sizeof(double)));
  cuda_safe_call(cudaMallocManaged(&Z, N * sizeof(double)));

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
    Z[i] = Y0(i);
  }

  double alpha = 3.14;
  double beta  = 1664.0;

  auto e = ctx.epoch();

  ctx.parallel_for(box(N), e)->*[alpha, X, Y] __device__(size_t i) {
    Y[i] += alpha * X[i];
  };

  ctx.parallel_for(box(N), e)->*[beta, X, Z] __device__(size_t i) {
    Z[i] += beta * X[i];
  };

  ++e;

  ctx.parallel_for(box(N), e)->*[Y, Z] __device__(size_t i) {
    Z[i] += Y[i];
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(X[i] - X0(i)) < 0.0001);
    assert(fabs(Z[i] - (Y0(i) + Y0(i) + alpha * X0(i) + beta * X0(i))) < 0.0001);
  }
}
