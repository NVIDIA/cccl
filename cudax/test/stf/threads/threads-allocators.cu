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
 * @brief Concurrent task submission on one stream_ctx with each block
 *        allocator policy (cached/default, pooled, buddy)
 *
 * The context allocator is shared by all logical data: task acquisition only
 * holds per-logical-data mutexes, so submitter threads with distinct logical
 * data reach the allocator concurrently. This exercises the allocator's lazy
 * per-place initialization and its block metadata under contention, mixing:
 *  - rw tasks on logical data shared between all threads (checked at the end:
 *    every element must equal the total number of rw tasks, i.e. STF
 *    serialized them), and
 *  - create/use/destroy churn of thread-private logical data (exercises the
 *    logical_data registry and deferred deallocation under contention).
 */

#include <cuda/experimental/__stf/allocators/buddy_allocator.cuh>
#include <cuda/experimental/__stf/allocators/pooled_allocator.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <atomic>
#include <thread>
#include <vector>

using namespace cuda::experimental::stf;

static __global__ void incr_kernel(slice<int> s)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  for (int i = tid; i < static_cast<int>(s.size()); i += nthreads)
  {
    s(i) += 1;
  }
}

enum class alloc_kind
{
  default_cached,
  pooled,
  buddy
};

static void run(alloc_kind kind)
{
  const size_t nthreads = 8;
  const size_t iters    = 32;
  const size_t nshared  = 4;
  const size_t N        = 1024;

  stream_ctx ctx;

  switch (kind)
  {
    case alloc_kind::pooled:
      ctx.set_allocator(block_allocator<pooled_allocator>(ctx));
      break;
    case alloc_kind::buddy:
      ctx.set_allocator(block_allocator<buddy_allocator>(ctx));
      break;
    default:
      break; // keep the default (cached) allocator
  }

  ::std::vector<logical_data<slice<int>>> shared;
  for (size_t s = 0; s < nshared; s++)
  {
    auto l = ctx.logical_data(shape_of<slice<int>>(N));
    ctx.task(l.write())->*[](cudaStream_t stream, auto sl) {
      cuda_safe_call(cudaMemsetAsync(sl.data_handle(), 0, sl.size() * sizeof(int), stream));
    };
    shared.push_back(mv(l));
  }

  ::std::vector<::std::atomic<int>> rw_count(nshared);
  for (auto& c : rw_count)
  {
    c.store(0);
  }

  ::std::atomic<size_t> n_ready{0};

  const auto worker = [&](const size_t tid) {
    // Rendezvous: no thread submits until all workers exist, so submissions
    // genuinely overlap instead of running serially as threads spawn.
    n_ready.fetch_add(1);
    while (n_ready.load() < nthreads)
    {
    }

    for (size_t i = 0; i < iters; i++)
    {
      const size_t target = (tid + i) % nshared;
      if ((tid + i) % 3 != 0)
      {
        // rw increment on a shared logical data
        ctx.task(shared[target].rw())->*[](cudaStream_t stream, auto sl) {
          incr_kernel<<<8, 128, 0, stream>>>(sl);
        };
        rw_count[target].fetch_add(1, ::std::memory_order_relaxed);
      }
      else
      {
        // churn: thread-private logical data created, used, destroyed. This
        // makes the allocator allocate and (asynchronously) deallocate under
        // contention, and exercises the logical_data id registry.
        auto lp = ctx.logical_data(shape_of<slice<int>>(256));
        ctx.task(lp.write())->*[](cudaStream_t stream, auto sl) {
          cuda_safe_call(cudaMemsetAsync(sl.data_handle(), 0, sl.size() * sizeof(int), stream));
        };
        ctx.task(lp.rw())->*[](cudaStream_t stream, auto sl) {
          incr_kernel<<<8, 128, 0, stream>>>(sl);
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

  for (size_t s = 0; s < nshared; s++)
  {
    const int expected = rw_count[s].load();
    ctx.host_launch(shared[s].read())->*[expected](auto sl) {
      for (size_t i = 0; i < sl.size(); i++)
      {
        EXPECT(sl(i) == expected);
      }
    };
  }

  ctx.finalize();
}

int main()
{
  run(alloc_kind::default_cached);
  run(alloc_kind::pooled);
  run(alloc_kind::buddy);
}
