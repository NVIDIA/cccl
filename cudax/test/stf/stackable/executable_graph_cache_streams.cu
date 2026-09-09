//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

#include <atomic>
#include <thread>

using namespace cuda::experimental::stf;

__global__ void keep_stream_busy(unsigned long long cycles)
{
  const unsigned long long start = clock64();
  while (clock64() - start < cycles)
  {
  }
}

void test_cache_stream_affinity()
{
  cudaStream_t stream_a = cuda_try<cudaStreamCreate>();
  cudaStream_t stream_b = cuda_try<cudaStreamCreate>();

  {
    async_resources_handle handle;
    cudaGraph_t graph_a = cuda_try<cudaGraphCreate>(0);
    cudaGraph_t graph_b = cuda_try<cudaGraphCreate>(0);

    cuda_try<cudaGraphAddEmptyNode>(graph_a, nullptr, 0);
    cuda_try<cudaGraphAddEmptyNode>(graph_b, nullptr, 0);

    auto [exec_a, hit_a] = handle.cached_graphs_query(1, 0, graph_a, stream_a);
    EXPECT(!hit_a);

    auto [exec_a_again, hit_a_again] = handle.cached_graphs_query(1, 0, graph_b, stream_a);
    EXPECT(hit_a_again);
    EXPECT(exec_a_again == exec_a);

    auto [exec_b, hit_b] = handle.cached_graphs_query(1, 0, graph_b, stream_b);
    EXPECT(!hit_b);
    EXPECT(exec_b != exec_a);

    keep_stream_busy<<<1, 1, 0, stream_a>>>(100000000);
    cuda_try(cudaPeekAtLastError());

    auto [exec_busy, hit_busy] = handle.cached_graphs_query(1, 0, graph_b, stream_a);
    EXPECT(!hit_busy);
    EXPECT(exec_busy != exec_a);

    cuda_try(cudaStreamSynchronize(stream_a));
    cuda_try(cudaGraphDestroy(graph_b));
    cuda_try(cudaGraphDestroy(graph_a));
  }

  cuda_try(cudaStreamDestroy(stream_b));
  cuda_try(cudaStreamDestroy(stream_a));
}

template <typename Data>
void build_scope(
  stackable_ctx& ctx,
  Data data,
  int value,
  int* active,
  int* max_active,
  ::std::atomic<int>& ready,
  ::std::atomic<bool>& launch)
{
  ctx.set_head_offset(ctx.get_root_offset());
  auto scope = ctx.graph_scope();
  ctx.parallel_for(data.shape(), data.write())->*[active, max_active, value] __device__(size_t i, auto data) {
    if (i == 0)
    {
      const int concurrent = atomicAdd(active, 1) + 1;
      atomicMax(max_active, concurrent);
      const unsigned long long start = clock64();
      while (clock64() - start < 200000000)
      {
      }
      atomicSub(active, 1);
    }
    data(i) = value;
  };

  ready.fetch_add(1, ::std::memory_order_release);
  while (!launch.load(::std::memory_order_acquire))
  {
  }
}

void test_sibling_graph_scopes_overlap()
{
  constexpr size_t count = 128;
  int output_a[count]    = {};
  int output_b[count]    = {};

  int* active     = nullptr;
  int* max_active = nullptr;
  cuda_try(cudaMallocManaged(&active, sizeof(int)));
  cuda_try(cudaMallocManaged(&max_active, sizeof(int)));
  *active     = 0;
  *max_active = 0;

  stackable_ctx ctx;
  auto data_a = ctx.logical_data(output_a);
  auto data_b = ctx.logical_data(output_b);

  ::std::atomic<int> ready{0};
  ::std::atomic<bool> launch{false};

  ::std::thread thread_a([&] {
    build_scope(ctx, data_a, 17, active, max_active, ready, launch);
  });
  ::std::thread thread_b([&] {
    build_scope(ctx, data_b, 29, active, max_active, ready, launch);
  });

  while (ready.load(::std::memory_order_acquire) != 2)
  {
  }
  launch.store(true, ::std::memory_order_release);

  thread_a.join();
  thread_b.join();

  ctx.finalize();

  EXPECT(*max_active == 2);
  for (size_t i = 0; i < count; ++i)
  {
    EXPECT(output_a[i] == 17);
    EXPECT(output_b[i] == 29);
  }

  cuda_try(cudaFree(max_active));
  cuda_try(cudaFree(active));
}

int main()
{
  test_cache_stream_affinity();
  test_sibling_graph_scopes_overlap();
}
