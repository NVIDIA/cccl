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
 * @brief Semantics of `place_memory_resource` stream-ordered allocation under
 *        CUDA graph capture. The shipped contract is "allocate outside
 *        capture; capture computation only" — but a capture-time allocation
 *        is RECORDED, not refused: it becomes a graph-owned allocation
 *        (mem-alloc node). This test pins the two shapes of that behavior:
 *
 *         - balanced allocate/deallocate enclosed in the capture (the CUB
 *           temporary-allocation shape) instantiates and replays freely;
 *         - an allocation NOT freed inside the graph makes an immediate
 *           relaunch fail predictably until the pointer is freed outside.
 */

#include <cuda/stream>

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;
using cuda::experimental::places::place_memory_resource;

namespace
{
__global__ void write_kernel(float* p, size_t n, float v)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    p[i] = v;
  }
}

__global__ void copy_kernel(const float* src, float* dst, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    dst[i] = src[i];
  }
}

bool graph_has_node_type(cudaGraph_t graph, cudaGraphNodeType type)
{
  size_t num = 0;
  cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &num));
  ::std::vector<cudaGraphNode_t> nodes(num);
  cuda_safe_call(cudaGraphGetNodes(graph, nodes.data(), &num));
  for (cudaGraphNode_t node : nodes)
  {
    cudaGraphNodeType t;
    cuda_safe_call(cudaGraphNodeGetType(node, &t));
    if (t == type)
    {
      return true;
    }
  }
  return false;
}

// Balanced alloc/free pair inside the capture: the shape CUB algorithm
// temporaries take. Becomes graph-owned memory; instantiates and replays.
void test_balanced_alloc_free_replays(place_group& group)
{
  const size_t n     = 1 << 16;
  const size_t bytes = n * sizeof(float);
  auto& shard_place  = group.place(0);
  cudaStream_t s     = group.get_stream(0);
  place_memory_resource mr(shard_place.affine_data_place());

  // Persistent output written by the captured kernels
  float* d_out = static_cast<float*>(mr.allocate(::cuda::stream_ref{s}, bytes));
  cuda_safe_call(cudaStreamSynchronize(s));

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  ::cuda::stream_ref{s}.wait(::cuda::stream_ref{origin});

  float* d_tmp = static_cast<float*>(mr.allocate(::cuda::stream_ref{s}, bytes)); // graph-owned
  {
    cuda::experimental::places::exec_place_scope scope(shard_place);
    write_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(d_tmp, n, 7.0f);
    cuda_safe_call(cudaGetLastError());
    copy_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(d_tmp, d_out, n);
    cuda_safe_call(cudaGetLastError());
  }
  mr.deallocate(::cuda::stream_ref{s}, d_tmp, bytes); // freed inside the graph

  ::cuda::stream_ref{origin}.wait(::cuda::stream_ref{s});
  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));

  // The allocation was recorded as a graph memory node
  EXPECT(graph_has_node_type(graph, cudaGraphNodeTypeMemAlloc));
  EXPECT(graph_has_node_type(graph, cudaGraphNodeTypeMemFree));

  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  ::std::vector<float> host(n);
  for (int round = 0; round < 3; round++)
  {
    cuda_safe_call(cudaGraphLaunch(exec, origin));
    cuda_safe_call(cudaStreamSynchronize(origin));
    cuda_safe_call(cudaMemcpy(host.data(), d_out, bytes, cudaMemcpyDefault));
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(host[i] == 7.0f);
    }
    cuda_safe_call(cudaMemset(d_out, 0, bytes)); // force the next replay to redo the work
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  mr.deallocate(::cuda::stream_ref{s}, d_out, bytes);
  cuda_safe_call(cudaStreamSynchronize(s));
  cuda_safe_call(cudaStreamDestroy(origin));
}

// Violating "allocate outside capture; free what you capture-allocate": an
// allocation left unfreed by the graph stays live after a launch, and an
// immediate relaunch fails predictably until it is freed outside the graph.
void test_unbalanced_alloc_fails_predictably(place_group& group)
{
  const size_t n     = 1 << 16;
  const size_t bytes = n * sizeof(float);
  auto& shard_place  = group.place(0);
  cudaStream_t s     = group.get_stream(0);
  place_memory_resource mr(shard_place.affine_data_place());

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  ::cuda::stream_ref{s}.wait(::cuda::stream_ref{origin});
  float* d_leak = static_cast<float*>(mr.allocate(::cuda::stream_ref{s}, bytes)); // NOT freed in-graph
  {
    cuda::experimental::places::exec_place_scope scope(shard_place);
    write_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(d_leak, n, 3.0f);
    cuda_safe_call(cudaGetLastError());
  }
  ::cuda::stream_ref{origin}.wait(::cuda::stream_ref{s});
  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph_has_node_type(graph, cudaGraphNodeTypeMemAlloc));
  EXPECT(!graph_has_node_type(graph, cudaGraphNodeTypeMemFree));

  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  // The allocation is live after the launch: the captured pointer reads back
  ::std::vector<float> host(n);
  cuda_safe_call(cudaMemcpy(host.data(), d_leak, bytes, cudaMemcpyDefault));
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 3.0f);
  }

  // Relaunching while the previous launch's allocation is still live fails
  const cudaError_t relaunch = cudaGraphLaunch(exec, origin);
  EXPECT(relaunch != cudaSuccess);
  (void) cudaGetLastError(); // clear

  // Freeing the graph allocation outside the graph re-arms the launch
  cuda_safe_call(cudaFreeAsync(d_leak, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));
  cuda_safe_call(cudaFreeAsync(d_leak, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_balanced_alloc_free_replays(group);
  test_unbalanced_alloc_fails_predictably(group);

  return 0;
}
