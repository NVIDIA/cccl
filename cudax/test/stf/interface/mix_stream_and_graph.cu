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
 *        stream backend
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

int main()
{
  stream_ctx ctx;
  constexpr int N = 12;

  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = i;
  }

  auto lX = ctx.logical_data(X);

  ctx.task(lX.rw())->*[](cudaStream_t stream, auto sX) {
    graph_ctx gctx;
    auto lX_alias = gctx.logical_data(sX, data_place::current_device());

    // X(i) = (i + 17)
    gctx.task(lX_alias.rw())->*[](cudaStream_t stream2, auto sX) {
      add<<<16, 128, 0, stream2>>>(sX, 17);
    };

    // X(i) = 2*(i + 17) + 1
    gctx.host_launch(lX_alias.rw())->*[&](auto sX) {
      for (int ind = 0; ind < N; ind++)
      {
        sX(ind) = 2 * sX(ind) + 1;
      }
    };

    gctx.submit(stream);
    // no sync !
  };

  ctx.finalize();

  for (int ind = 0; ind < N; ind++)
  {
    EXPECT(X[ind] == 2 * (ind + 17) + 1);
  }
}
