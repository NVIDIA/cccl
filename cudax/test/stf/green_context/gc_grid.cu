//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/__stf/places/place_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
#if CUDA_VERSION < 12040
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else
  stream_ctx ctx;

  int NITER   = 8;
  const int n = 16 * 1024 * 1024;

  std::vector<double> X(n);
  std::vector<double> Y(n);

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));
  auto handle_Y = ctx.logical_data(make_slice(&Y[0], n));

  auto where = place_partition(ctx.async_resources(), exec_place::all_devices(), place_partition_scope::green_context).as_grid();

  for (int iter = 0; iter < NITER; iter++)
  {
    ctx.parallel_for(blocked_partition(), where, handle_X.shape(), handle_X.rw(), handle_Y.read())
        ->*[] __device__(size_t i, auto x, auto y) {
              x(i) += y(i);
            };
  }

  ctx.finalize();
#endif // CUDA_VERSION < 12040
}
