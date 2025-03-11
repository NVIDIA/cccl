//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

#include <functional>
#include <thread>

using namespace cuda::experimental::stf;

void worker(
  stream_ctx ctx, int id, frozen_logical_data<slice<int>> fAi, frozen_logical_data<slice<int>> fB, ::std::mutex& mutex)
{
  cudaStream_t stream = ctx.pick_stream();

  auto gctx = graph_ctx(stream);

  mutex.lock();
  auto dAi = fAi.get(data_place::current_device(), stream);
  auto dB  = fB.get(data_place::current_device(), stream);
  mutex.unlock();

  auto g_lAi = gctx.logical_data(dAi, data_place::current_device());
  auto g_lB  = gctx.logical_data(dB, data_place::current_device());

  gctx.parallel_for(g_lAi.shape(), g_lAi.rw(), g_lB.read())->*[id] __device__(size_t j, auto ai, auto b) {
    ai(j) += id + b(j);
  };

  gctx.finalize();
}

int main()
{
  ::std::mutex mutex;
  stream_ctx ctx;

  const int N = 128000;

  const int NTHREADS = 8;

  // A vector of per-thread vectors
  ::std::vector<::std::vector<int>> A(NTHREADS);
  ::std::vector<logical_data<slice<int>>> lA(NTHREADS);

  for (int i = 0; i < NTHREADS; i++)
  {
    A[i].resize(N);
    for (int j = 0; j < N; j++)
    {
      A[i][j] = (i + j);
    }

    lA[i] = ctx.logical_data(A[i].data(), {N}).set_symbol("A_" + ::std::to_string(i));
  }

  ::std::vector<int> B(N);
  logical_data<slice<int>> lB;
  for (int j = 0; j < N; j++)
  {
    B[j] = (17 * j + 3);
  }
  lB = ctx.logical_data(B.data(), {N}).set_symbol("B");

  ::std::vector<frozen_logical_data<slice<int>>> fA(NTHREADS);
  for (int i = 0; i < NTHREADS; i++)
  {
    fA[i] = ctx.freeze(lA[i], access_mode::rw, data_place::current_device());
    fA[i].set_automatic_unfreeze(true);
  }

  auto fB = ctx.freeze(lB, access_mode::read, data_place::current_device());
  fB.set_automatic_unfreeze(true);

  ::std::vector<::std::thread> threads;
  for (int i = 0; i < NTHREADS; ++i)
  {
    threads.emplace_back(worker, ctx, i, fA[i], fB, ::std::ref(mutex));
  }

  cudaStream_t stream = ctx.pick_stream();

  for (int i = 0; i < NTHREADS; ++i)
  {
    threads[i].join();
    fA[i].unfreeze(stream);
  }

  fB.unfreeze(stream);

  for (int i = 0; i < NTHREADS; ++i)
  {
    ctx.host_launch(lA[i].read())->*[i](auto ai) {
      for (size_t j = 0; j < N; j++)
      {
        EXPECT(ai(j) == (i + j) + (i + (17 * j + 3)));
      }
    };
  }

  ctx.finalize();
}
