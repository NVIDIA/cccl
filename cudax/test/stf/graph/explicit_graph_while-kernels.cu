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
 * @brief Add tasks to a user-provided child graph from a while loop
 *
 * Adapted from https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)
__global__ void dummy() {}

__device__ int counter = 5;

__global__ void setHandle(cudaGraphConditionalHandle handle)
{
  unsigned int value = 0;
  // We could perform some work here and set value based on the result of that work.
  if (counter-- > 0)
  {
    // Set ‘value’ to non-zero if we want the conditional body to execute
    value = 1;
  }
  cudaGraphSetConditional(handle, value);
}
#endif // _CCCL_CTK_AT_LEAST(12, 4)

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
#else
  cudaStream_t stream;

  cuda_safe_call(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaGraph_t graph;
  cudaGraphNode_t kernelNode, conditionalNode;
  void* kernelArgs[1];

  cudaGraphCreate(&graph, 0);

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph);

  // Use a kernel upstream of the conditional to set the handle value
  cudaGraphNodeParams kParams = {};
  kParams.type                = cudaGraphNodeTypeKernel;
  kParams.kernel.func         = (void*) setHandle;
  kParams.kernel.gridDim.x = kParams.kernel.gridDim.y = kParams.kernel.gridDim.z = 1;
  kParams.kernel.blockDim.x = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;
  kParams.kernel.kernelParams                                                       = kernelArgs;
  kernelArgs[0]                                                                     = &handle;
  cudaGraphAddNode(&kernelNode, graph, NULL, 0, &kParams);

  cudaGraphNodeParams cParams = {};
  cParams.type                = cudaGraphNodeTypeConditional;
  cParams.conditional.handle  = handle;
  cParams.conditional.type    = cudaGraphCondTypeSwitch;
  cParams.conditional.size    = 3;
  cudaGraphAddNode(&conditionalNode, graph, &kernelNode, 1, &cParams);

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  graph_ctx ctx(bodyGraph);

  auto lX = ctx.token();
  auto lY = ctx.token();
  auto lZ = ctx.token();

  ctx.cuda_kernel(lX.write())->*[]() {
    return cuda_kernel_desc{dummy, 1, 1, 0};
  };

  ctx.cuda_kernel(lX.read(), lY.write())->*[]() {
    return cuda_kernel_desc{dummy, 1, 1, 0};
  };

  ctx.cuda_kernel(lX.read(), lZ.write())->*[]() {
    return cuda_kernel_desc{dummy, 1, 1, 0};
  };

  ctx.cuda_kernel(lY.rw(), lZ.rw())->*[]() {
    return cuda_kernel_desc{dummy, 1, 1, 0};
  };

  ctx.finalize_as_graph();

  cudaGraphExec_t graphExec = NULL;
  cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  cuda_safe_call(cudaGraphLaunch(graphExec, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));
  cuda_safe_call(cudaGraphDebugDotPrint(graph, "test-while.dot", cudaGraphDebugDotFlags(0)));
#endif // !_CCCL_CTK_BELOW(12, 4)
}
