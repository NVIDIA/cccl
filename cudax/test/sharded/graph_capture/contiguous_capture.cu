//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Graph capture over a contiguous (`allocate_contiguous`) sharded
 *        array: the VMM mappings pre-exist the capture, so the backing is
 *        transparent to the graph — per-shard elementwise stages and a
 *        whole-array kernel through `contiguous_data()` capture into ONE
 *        graph and replay with inputs mutated between launches.
 */

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place_scope;
using cuda::experimental::places::place_group;

namespace
{
struct triple_op
{
  __host__ __device__ float operator()(float x) const
  {
    return 3.0f * x;
  }
};

// Whole-array stage: an unmodified single-pointer kernel over the ONE
// contiguous VA range
__global__ void plus_one_all(float* base, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    base[i] += 1.0f;
  }
}

void test_contiguous_pipeline_capture(place_group& group)
{
  const size_t n = (1 << 21) + 37; // > one 2 MiB granule per shard, uneven
  auto data      = sharded_array<float>::allocate_contiguous(group, n);
  EXPECT(data.is_contiguous());
  float* base = data.contiguous_data();
  EXPECT(base != nullptr);

  fill(group, data, 1.0f); // warm-up outside capture

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  // Capture: per-shard transform (on the shard streams), join, then the
  // whole-array kernel on the origin stream through the base pointer
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  data.fork_from(origin);
  transform(group, data, triple_op{}, /*blocking=*/false);
  data.join_into(origin);
  plus_one_all<<<static_cast<unsigned>((n + 255) / 256), 256, 0, origin>>>(base, n);
  cuda_safe_call(cudaGetLastError());

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));

  // Replay with inputs mutated between launches: data' = 3 * data + 1
  ::std::vector<float> host(n);
  for (int round = 0; round < 3; round++)
  {
    const float v = static_cast<float>(round + 1);
    fill(group, data, v); // outside the graph
    cuda_safe_call(cudaGraphLaunch(exec, origin));
    cuda_safe_call(cudaStreamSynchronize(origin));

    // Read back through the contiguous base pointer, as one plain array
    cuda_safe_call(cudaMemcpy(host.data(), base, n * sizeof(float), cudaMemcpyDefault));
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(host[i] == 3.0f * v + 1.0f);
    }
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_contiguous_pipeline_capture(group);

  return 0;
}
