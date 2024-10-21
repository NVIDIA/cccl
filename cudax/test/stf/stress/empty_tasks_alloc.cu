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
    auto lX = ctx.logical_data(shape_of<slice<double>>(N));
    auto lY = ctx.logical_data(shape_of<slice<double>>(N));
    ctx.task(lX.write())->*[](cudaStream_t, auto) {};
    ctx.task(lY.write())->*[](cudaStream_t, auto) {};
    ctx.task(lX.read(), lY.rw())->*[](cudaStream_t, auto, auto) {};
  }
  stop = std::chrono::steady_clock::now();
  ctx.finalize();

  std::chrono::duration<double> duration = stop - start;
  fprintf(stderr, "Elapsed: %.2lf us per task\n", duration.count() * 1000000.0 / (3 * iter_cnt));
}
