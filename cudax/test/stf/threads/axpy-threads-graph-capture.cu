//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Ensure a graph_ctx can be used concurrently
 *
 */

#include <cuda/experimental/stf.cuh>

#include <mutex>
#include <thread>

using namespace cuda::experimental::stf;

__global__ void axpy(slice<int> y, slice<const int> x, int a)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < x.size(); i += nthreads)
  {
    y(i) += a * x(i);
  }
}

void mytask(graph_ctx ctx, int /*id*/)
{
  const size_t N = 16;

  int alpha = 3;

  auto lX = ctx.logical_data<int>(N);
  auto lY = ctx.logical_data<int>(N);

  ctx.parallel_for(lX.shape(), lX.write(), lY.write())->*[] __device__(size_t i, auto x, auto y) {
    x(i) = (1 + i);
    y(i) = (2 + i * i);
  };

  /* Compute Y = Y + alpha X */
  for (size_t i = 0; i < 200; i++)
  {
    ctx.task(lY.rw(), lX.read())->*[alpha](cudaStream_t stream, auto dY, auto dX) {
      axpy<<<128, 64, 0, stream>>>(dY, dX, alpha);
    };
  }

  ctx.host_launch(lX.read(), lY.read())->*[alpha](auto x, auto y) {
    for (size_t i = 0; i < N; i++)
    {
      EXPECT(x(i) == 1 + i);
      EXPECT(y(i) == 2 + i * i + 200 * alpha * x(i));
    }
  };
}

int main()
{
  graph_ctx ctx;

  ::std::vector<::std::thread> threads;
  // Launch threads
  for (int i = 0; i < 10; ++i)
  {
    threads.emplace_back(mytask, ctx, i);
  }

  // Wait for all threads to complete.
  for (auto& th : threads)
  {
    th.join();
  }

  ctx.finalize();
}
