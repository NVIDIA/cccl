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

#include <chrono>

using namespace std::chrono;
using namespace cuda::experimental::stf;

/* wall-clock time */
double gettime()
{
  auto now = system_clock::now().time_since_epoch();
  return duration_cast<duration<double>>(now).count();
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

  size_t nloops       = 10;
  size_t inner_nloops = 1000;

  if (argc > 1)
  {
    nloops = atol(argv[1]);
  }

  if (argc > 2)
  {
    inner_nloops = atol(argv[2]);
  }

  for (size_t j = 0; j < nloops; j++)
  {
    std::chrono::steady_clock::time_point start, stop;
    start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < inner_nloops; i++)
    {
      ctx.task(lX.read(), lY.rw())->*[](cudaStream_t, auto, auto) {};
    }

    stop                                   = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = stop - start;

    fprintf(stderr, "Elapsed: %.2lf us per task\n", duration.count() * 1000000.0 / (inner_nloops));
  }

  ctx.finalize();
}
