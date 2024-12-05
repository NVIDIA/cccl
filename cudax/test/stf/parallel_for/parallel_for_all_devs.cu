//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void axpy(size_t start, size_t cnt, T a, const T* x, T* y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int ind = tid; ind < cnt; ind += nthreads)
  {
    y[ind + start] += a * x[ind + start];
  }
}

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
  stream_ctx ctx;

  const int N = 1024 * 1024 * 32;
  double *X, *Y;

  X = new double[N];
  Y = new double[N];

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = X0(ind);
    Y[ind] = Y0(ind);
  }

  auto handle_X = ctx.logical_data(X, {N});
  auto handle_Y = ctx.logical_data(Y, {N});

  auto all_devs = exec_place::all_devices();

  double alpha = 3.14;

  /* Compute Y = Y + alpha X */
  ctx.parallel_for(tiled_partition<1024 * 1024>(), all_devs, handle_X.shape(), handle_X.read(), handle_Y.rw())
      ->*[=] _CCCL_DEVICE(size_t i, auto sX, auto sY) {
            sY(i) += alpha * sX(i);
          };

  /* Check the result on the host */
  ctx.task(exec_place::host, handle_X.read(), handle_Y.read())->*[&](cudaStream_t s, auto sX, auto sY) {
    cuda_safe_call(cudaStreamSynchronize(s));

    for (size_t ind = 0; ind < N; ind++)
    {
      // Y should be Y0 + alpha X0
      EXPECT(fabs(sY(ind) - (Y0(ind) + alpha * X0(ind))) < 0.0001);

      // X should be X0
      EXPECT(fabs(sX(ind) - X0(ind)) < 0.0001);
    }
  };

  ctx.finalize();
}
