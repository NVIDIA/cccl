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
 * @brief Test that we can use a local graph context within a task of the
 *        stream backend to create a CUDA graph that we can launch multiple
 *        times.
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
__global__ void setup(slice<T> s)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t ind = tid; ind < s.size(); ind += nthreads)
  {
    s(ind) = T(ind);
  }
}

template <typename T>
__global__ void add(slice<T> s, T val)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t ind = tid; ind < s.size(); ind += nthreads)
  {
    s(ind) += val;
  }
}

__global__ void slice_add(slice<const int> s_from, slice<int> s_to)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t ind = tid; ind < s_from.size(); ind += nthreads)
  {
    s_to(ind) += s_from(ind);
  }
}

int main()
{
  stream_ctx ctx;
  const int N    = 12;
  const size_t K = 10;

  int X[N];
  int Y[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = i;
    Y[i] = -i;
  }

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  /*
   * Create a CUDA graph from a single task, and launch it many times
   */

  ctx.task(lX.rw(), lY.rw())->*[&](cudaStream_t stream, auto sX, auto sY) {
    graph_ctx gctx;
    auto lX_alias = gctx.logical_data(sX, data_place::current_device());
    auto lY_alias = gctx.logical_data(sY, data_place::current_device());

    for (size_t ii = 0; ii < 10; ii++)
    {
      gctx.task(lX_alias.rw())->*[](cudaStream_t stream2, auto sX) {
        add<<<16, 128, 0, stream2>>>(sX, 17);
      };
      gctx.task(lY_alias.rw())->*[](cudaStream_t stream2, auto sY) {
        add<<<16, 128, 0, stream2>>>(sY, 17);
      };
      gctx.task(lX_alias.read(), lY_alias.rw())->*[](cudaStream_t stream2, auto sX, auto sY) {
        slice_add<<<16, 128, 0, stream2>>>(sX, sY);
      };
      gctx.task(lX_alias.rw())->*[](cudaStream_t stream2, auto sX) {
        add<<<16, 128, 0, stream2>>>(sX, 17);
      };
      gctx.task(lY_alias.rw())->*[](cudaStream_t stream2, auto sY) {
        add<<<16, 128, 0, stream2>>>(sY, 17);
      };
    }

    //        gctx.host_launch(lX_alias.rw())->*[&](auto sX) {
    //            for (size_t ind = 0; ind < N; ind++) {
    //                sX(ind) = 2 * sX(ind) + 1;
    //            }
    //        };

    // gctx.print_to_dot("gctx" + std::to_string(iter));
    auto exec_graph = gctx.instantiate();
    for (size_t iter = 0; iter < K; iter++)
    {
      cudaGraphLaunch(*exec_graph, stream);
    }
  };

  ctx.finalize();
}
