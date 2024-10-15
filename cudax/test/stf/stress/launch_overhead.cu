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
  int iter_cnt = 1000000;
#else
  int iter_cnt = 10000;
  fprintf(stderr, "Warning: Running with small problem size in debug mode, should use DEBUG=0.\n");
#endif

  if (argc > 1)
  {
    iter_cnt = atoi(argv[1]);
  }

  std::chrono::steady_clock::time_point start, stop;
  start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < iter_cnt; iter++)
  {
    ctx.launch(lX.read(), lY.rw())->*[] _CCCL_DEVICE(auto th, auto X, auto Y) {
      for (size_t i = th.rank(); i < X.size(); i += th.size())
      {
        Y(i) = 2.0 * X(i);
      }
    };

    ctx.launch(lX.rw(), lY.read())->*[] _CCCL_DEVICE(auto th, auto X, auto Y) {
      for (size_t i = th.rank(); i < X.size(); i += th.size())
      {
        X(i) = 0.5 * Y(i);
      }
    };
  }
  stop = std::chrono::steady_clock::now();
  ctx.finalize();

  std::chrono::duration<double> duration = stop - start;
  fprintf(stderr, "Elapsed: %.2lf us per task\n", duration.count() * 1000000.0 / (2 * iter_cnt));
}
