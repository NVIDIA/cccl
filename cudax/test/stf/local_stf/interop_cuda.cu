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
 * @brief This example illustrates how to introduce CUDASTF contexts within existing stream-synchronized library calls
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// B += alpha*A;
__global__ void axpy(double alpha, double* d_ptrA, double* d_ptrB, size_t N)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < N; i += nthreads)
  {
    d_ptrB[i] += alpha * d_ptrA[i];
  }
}

int main()
{
  double *d_ptrA, *d_ptrB;
  const size_t N     = 128 * 1024;
  const size_t NITER = 128;

  // User allocated memory
  cuda_safe_call(cudaMalloc(&d_ptrA, N * sizeof(double)));
  cuda_safe_call(cudaMalloc(&d_ptrB, N * sizeof(double)));

  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  async_resources_handle handle;
  for (size_t i = 0; i < NITER; i++)
  {
    stream_ctx ctx(stream, handle);
    auto lA = ctx.logical_data(make_slice(d_ptrA, N), data_place::current_device());
    auto lB = ctx.logical_data(make_slice(d_ptrB, N), data_place::current_device());

    ctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
      a(i) = sin(double(i));
    };
    ctx.parallel_for(lB.shape(), lB.write())->*[] __device__(size_t i, auto b) {
      b(i) = cos(double(i));
    };

    ctx.task(lA.read(), lB.rw())->*[=](cudaStream_t s, auto a, auto b) {
      axpy<<<128, 32, 0, s>>>(3.0, a.data_handle(), b.data_handle(), N);
    };

    // Note that this is non-blocking because we have creating the stream_ctx
    // relative to a user-provided CUDA stream
    ctx.finalize();

    axpy<<<128, 32, 0, stream>>>(2.0, d_ptrA, d_ptrB, N);
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
}
