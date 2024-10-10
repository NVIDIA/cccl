//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void swap(slice<double> dst, const slice<double> src)
{
  size_t tid      = threadIdx.x + blockIdx.x * blockDim.x;
  size_t nthreads = blockDim.x * gridDim.x;
  size_t n        = dst.size();

  for (size_t i = tid; i < n; i += nthreads)
  {
    double tmp = dst(i);
    dst(i)     = src(i);
    src(i)     = tmp;
  }
}

int main(int argc, char** argv)
{
  stream_ctx ctx;
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = 1.0;
    Y[i] = 2.0;
  }

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

#ifdef NDEBUG
  size_t iter_cnt = 10000000;
#else
  size_t iter_cnt = 10000;
  fprintf(stderr, "Warning: Running with small problem size in debug mode, should use DEBUG=0.\n");
#endif

  if (argc > 1)
  {
    iter_cnt = atol(argv[1]);
  }

  std::chrono::steady_clock::time_point start, stop;
  start = std::chrono::steady_clock::now();
  for (size_t iter = 0; iter < iter_cnt; iter++)
  {
    ctx.task(lX.read(), lY.rw())->*[&](cudaStream_t s, auto dX, auto dY) {
      swap<<<4, 16, 0, s>>>(dY, dX);
    };
    ctx.task(lY.read(), lX.rw())->*[&](cudaStream_t s, auto dY, auto dX) {
      swap<<<4, 16, 0, s>>>(dX, dY);
    };
  }
  stop = std::chrono::steady_clock::now();
  ctx.finalize();

  std::chrono::duration<double> duration = stop - start;
  fprintf(stderr, "Elapsed: %.2lf us per task pair\n", duration.count() * 1000000.0 / (iter_cnt));
}
