//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void scal(size_t n, T a, T* x)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    x[ind] = a * x[ind];
  }
}

double x_init(int i)
{
  return cos((double) i);
}

template <typename Ctx>
void run()
{
  Ctx ctx;
  const int n = 4096;
  double X[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = x_init(ind);
  }

  auto handle_X = ctx.logical_data(X);

  double alpha = 2.0;
  int niter    = 4;
  for (int iter = 0; iter < niter; iter++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto sX) {
      scal<<<16, 128, 0, s>>>(sX.size(), alpha, sX.data_handle());
    };
  }

  // Ask to use Y on the host
  ctx.host_launch(handle_X.read())->*[&](auto sX) {
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(sX(ind) - pow(alpha, niter) * (x_init(ind))) < 0.00001);
    }
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
