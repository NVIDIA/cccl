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
//! \brief Test ctx_resource management with finalize_as_graph

#include <cuda/experimental/__stf/internal/context.cuh>
#include <atomic>

using namespace cuda::experimental::stf;

int main()
{
  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  context ctx = graph_ctx();  // Generic context holding a graph_ctx
  std::atomic<int> callback_count{0};

  // Add some work that will create resources (host_launch creates host callback args)
  ctx.host_launch()->*[&callback_count]() {
    callback_count.fetch_add(1);
  };

  // Get the graph without finalizing the context (access graph-specific method)
  ::std::shared_ptr<cudaGraph_t> graph = ctx.to_graph_ctx().finalize_as_graph();

  // Instantiate the graph
  cudaGraphExec_t graphExec;
  cuda_safe_call(cudaGraphInstantiate(&graphExec, *graph, nullptr, nullptr, 0));

  // Execute the graph multiple times
  for (int i = 0; i < 3; i++)
  {
    cuda_safe_call(cudaGraphLaunch(graphExec, stream));
  }

  // Clean up resources after graph execution using the generic context API
  ctx.release_resources(stream);  // Works with generic context!
  cuda_safe_call(cudaStreamSynchronize(stream));
  
  // Verify the callback was executed exactly 3 times
  EXPECT(callback_count.load() == 3);

  // Clean up
  cuda_safe_call(cudaGraphExecDestroy(graphExec));
  cuda_safe_call(cudaStreamDestroy(stream));

  return 0;
}
