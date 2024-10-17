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
 * @brief An AXPY kernel using an exec place attached to a specific CUDA stream
 *
 */

#include <cuda/experimental/stf.cuh>

#include "nvtx3/nvToolsExtCudaRt.h"

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
  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));
  nvtxNameCudaStreamA(stream, "user stream");

  // context ctx;
  stream_ctx ctx;
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
  auto where = exec_place::cuda_stream(stream);

  for (size_t iter = 0; iter < 10; iter++)
  {
    ctx.parallel_for(where, lX.shape(), lX.read(), lY.rw())->*[alpha] __device__(size_t i, auto x, auto y) {
      y(i) += alpha * x(i);
    };
  }

  /* Associate the CUDA stream with a unique internal ID to speed up synchronizations */
  auto rstream = register_stream(ctx.async_resources(), stream);
  auto where2  = exec_place::cuda_stream(rstream);

  for (size_t iter = 0; iter < 10; iter++)
  {
    ctx.parallel_for(where2, lX.shape(), lX.read(), lY.rw())->*[alpha] __device__(size_t i, auto x, auto y) {
      y(i) += alpha * x(i);
    };
  }

  // Remove the association
  unregister_stream(ctx.async_resources(), rstream);

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + 2 * 10.0 * alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }

  cuda_safe_call(cudaStreamDestroy(stream));
}
