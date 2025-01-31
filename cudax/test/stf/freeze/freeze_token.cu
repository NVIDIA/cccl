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
 * @brief Freeze data in read-only fashion
 *
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

int X0(int i)
{
  return 17 * i + 45;
}

__global__ void print(slice<int> s)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < s.size(); i += nthreads)
  {
    printf("%d %d\n", i, s(i));
  }
}

int main()
{
  stream_ctx ctx;

  cudaStream_t stream = ctx.pick_stream();

  const int N = 16;
  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  auto lX     = ctx.logical_data(X).set_symbol("X");
  auto ltoken = ctx.logical_token().set_symbol("token");

  ctx.parallel_for(lX.shape(), lX.rw(), ltoken.write()).set_symbol("X=2X")->*
    [] __device__(size_t i, auto x, void_interface) {
      x(i) *= 2;
    };

  auto ftoken = ctx.freeze(ltoken);

  [[maybe_unused]] void_interface dummy = ftoken.get(data_place::current_device(), stream);
  ftoken.unfreeze(stream);

  ctx.finalize();
}
