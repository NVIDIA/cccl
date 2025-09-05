//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Test resource management for graphs launched multiple times

#include <cuda/experimental/__stf/internal/context.cuh>

#include <atomic>

using namespace cuda::experimental::stf;

int main()
{
  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Test: Create a reusable graph that can be launched multiple times
  // while properly managing resources across multiple executions
  context ctx = graph_ctx(); // Generic context holding a graph_ctx
  std::atomic<int> callback_count{0};

  // Add work that creates resources (host_launch creates host callback resources)
  ctx.host_launch()->*[&callback_count]() {
    callback_count.fetch_add(1);
  };

  // Get reusable graph without finalizing context
  ::std::shared_ptr<cudaGraph_t> graph = ctx.to_graph_ctx().finalize_as_graph();

  // Instantiate the graph for multiple launches
  cudaGraphExec_t graphExec;
  cuda_safe_call(cudaGraphInstantiate(&graphExec, *graph, nullptr, nullptr, 0));

  // Launch the same graph multiple times - resources should be reused properly
  const int num_launches = 3;
  for (int i = 0; i < num_launches; i++)
  {
    cuda_safe_call(cudaGraphLaunch(graphExec, stream));
  }

  // Clean up resources after all graph executions
  ctx.release_resources(stream);
  cuda_safe_call(cudaStreamSynchronize(stream));

  // Verify the callback was executed once per graph launch
  EXPECT(callback_count.load() == num_launches);

  // Clean up
  cuda_safe_call(cudaGraphExecDestroy(graphExec));
  cuda_safe_call(cudaStreamDestroy(stream));

  return 0;
}
