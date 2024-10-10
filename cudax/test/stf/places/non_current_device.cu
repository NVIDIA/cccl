//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void axpy(int n, T a, const T* x, T* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < n; ind += nthreads)
  {
    y[ind] += a * x[ind];
  }
}

int main()
{
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  cuda_safe_call(cudaSetDevice(0));

  if (ndevs < 2)
  {
    fprintf(stderr, "Skipping test that needs at last 2 devices.\n");
    return 0;
  }

  stream_ctx ctx;

  const double alpha = 2.0;
  const int n        = 12;

  double X[n], Y[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  auto handle_X = ctx.logical_data(X);
  auto handle_Y = ctx.logical_data(Y);

  /* Compute Y = Y + alpha X, but leave X on the host and access it with mapped memory */
  ctx.task(exec_place::device(1), handle_X.read(), handle_Y.rw())->*[&](cudaStream_t stream, auto X, auto Y) {
    axpy<<<16, 128, 0, stream>>>(n, alpha, X.data_handle(), Y.data_handle());
  };

  // Access Ask to use X, Y and Z on the host
  ctx.task(exec_place::host, handle_X.read(), handle_Y.read())->*[&](cudaStream_t stream, auto X, auto Y) {
    cuda_safe_call(cudaStreamSynchronize(stream));

    for (int ind = 0; ind < n; ind++)
    {
      // X unchanged
      EXPECT(fabs(X(ind) - 1.0 * ind) < 0.00001);
      // Y = Y + alpha X
      EXPECT(fabs(Y(ind) - (-3.0 + ind * (2.0 + alpha))) < 0.00001);
    }
  };

  ctx.finalize();

  return 0;
}
