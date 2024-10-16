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

  const int N = 16;
  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  auto lX = ctx.logical_data(X).set_symbol("X");
  auto lY = ctx.logical_data(lX.shape()).set_symbol("Y");

  ctx.parallel_for(lX.shape(), lX.rw()).set_symbol("X=2X")->*[] __device__(size_t i, auto x) {
    x(i) *= 2;
  };

  // test 1 : implicit sync of gets
  {
    auto fx      = ctx.freeze(lX);
    auto [dX, _] = fx.get(data_place::current_device());

    // the stream returned by task_fence should depend on the get operation
    auto stream2 = ctx.task_fence();
    print<<<8, 4, 0, stream2>>>(dX);

    ctx.parallel_for(lX.shape(), lX.read(), lY.write()).set_symbol("Y=X")->*[] __device__(size_t i, auto x, auto y) {
      y(i) = x(i);
    };

    fx.unfreeze(stream2);
  }

  // test 2 : unfreeze with no events due to user sync
  {
    auto fx      = ctx.freeze(lX);
    auto [dX, _] = fx.get(data_place::current_device());

    // the stream returned by task_fence should depend on the get operation
    auto stream2 = ctx.task_fence();
    print<<<8, 4, 0, stream2>>>(dX);

    // We synchronize so there is nothing to depend on anymore
    cudaStreamSynchronize(stream2);
    fx.unfreeze(event_list());
  }

  ctx.parallel_for(lX.shape(), lX.rw()).set_symbol("X+=1")->*[] __device__(size_t i, auto x) {
    x(i) += 1;
  };

  ctx.finalize();

  for (int i = 0; i < N; i++)
  {
    EXPECT(X[i] == 2 * X0(i) + 1);
  }
}
