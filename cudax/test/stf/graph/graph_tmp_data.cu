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
 * @brief Ensure temporary data are destroyed
 *
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

__global__ void dummy() {}

int main()
{
  //    stream_ctx ctx;
  graph_ctx ctx;
  const int N = 16;
  double X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  auto lX = ctx.logical_data(X);

  for (int i = 0; i < 10; i++)
  {
    // fprintf(stderr, "START loop %ld\n", i);
    auto lY = ctx.logical_data(lX.shape());
    lY.set_symbol("tmp" + std::to_string(i));

    // fprintf(stderr, "START pfor %ld\n", i);
    ctx.parallel_for(lX.shape(), lX.rw(), lY.write())->*[] _CCCL_DEVICE(size_t ind, auto dX, auto dY) {
      dY(ind) = dX(ind);
      dX(ind) = dY(ind) + 1.0;
    };
    // fprintf(stderr, "End loop %ld\n", i);
  }
  // fprintf(stderr, "OVER...\n");

  ctx.finalize();
}
