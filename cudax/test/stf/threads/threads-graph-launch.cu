//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Concurrent ctx.launch() and host-place parallel_for on one graph_ctx
 *
 * All tasks of a graph_ctx insert nodes into ONE shared cudaGraph_t, and the
 * CUDA graph construction API is not thread-safe per graph, so every insertion
 * must hold the context's graph mutex. The device parallel_for / cuda_kernel /
 * host_launch paths always did; ctx.launch() and parallel_for on
 * exec_place::host() did not, so concurrent submission through them corrupted
 * the graph. This test drives both fixed paths from several threads (each
 * thread on private logical data, so the only shared state is the graph
 * itself) and checks the computed values.
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

#include <atomic>
#include <thread>
#include <vector>

using namespace cuda::experimental::stf;

int main()
{
  const size_t nthreads = 8;
  const size_t iters    = 32;
  const size_t N        = 256;

  graph_ctx ctx;

  ::std::vector<logical_data<slice<int>>> data;
  for (size_t t = 0; t < nthreads; t++)
  {
    auto l = ctx.logical_data(shape_of<slice<int>>(N));
    ctx.parallel_for(l.shape(), l.write())->*[] __device__(size_t i, auto s) {
      s(i) = 0;
    };
    data.push_back(mv(l));
  }

  ::std::atomic<size_t> n_ready{0};

  const auto worker = [&](const size_t tid) {
    // Rendezvous: no thread submits until all workers exist, so graph-node
    // insertions genuinely overlap instead of running serially as threads
    // spawn.
    n_ready.fetch_add(1);
    while (n_ready.load() < nthreads)
    {
    }

    for (size_t i = 0; i < iters; i++)
    {
      if (i % 2 == 0)
      {
        // ctx.launch : adds a kernel node to the shared context graph
        ctx.launch(data[tid].rw())->*[] __device__(auto th, auto s) {
          for (auto i : th.apply_partition(shape(s)))
          {
            s(i) += 1;
          }
        };
      }
      else
      {
        // parallel_for on the host place : adds a host node to the shared
        // context graph
        ctx.parallel_for(exec_place::host(), data[tid].shape(), data[tid].rw())->*[](size_t i, auto s) {
          s(i) += 1;
        };
      }
    }
  };

  ::std::vector<::std::thread> threads;
  for (size_t t = 0; t < nthreads; t++)
  {
    threads.emplace_back(worker, t);
  }
  for (auto& th : threads)
  {
    th.join();
  }

  for (size_t t = 0; t < nthreads; t++)
  {
    ctx.host_launch(data[t].read())->*[iters](auto s) {
      for (size_t i = 0; i < s.size(); i++)
      {
        EXPECT(s(i) == static_cast<int>(iters));
      }
    };
  }

  ctx.finalize();
}
