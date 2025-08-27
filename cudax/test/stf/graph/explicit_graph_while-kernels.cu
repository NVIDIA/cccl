//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! @file
//! @brief Add tasks to a user-provided child graph from a while loop

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)
__global__ void dummy() {}

__global__ void setHandle(cudaGraphConditionalHandle handle)
{
  static int count = 5;
  cudaGraphSetConditional(handle, --count ? 1 : 0);
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
  cudaGraphNode_t conditionalNode;

  cudaGraphCreate(&graph, 0);

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

  cudaGraphNodeParams cParams = {};
  cParams.type                = cudaGraphNodeTypeConditional;
  cParams.conditional.handle  = handle;
  cParams.conditional.type    = cudaGraphCondTypeWhile;
  cParams.conditional.size    = 1;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cudaGraphAddNode(&conditionalNode, graph, nullptr, nullptr, 0, &cParams);
#  else
  cudaGraphAddNode(&conditionalNode, graph, nullptr, 0, &cParams);
#  endif

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

  ctx.cuda_kernel()->*[handle]() {
    return cuda_kernel_desc{setHandle, 1, 1, 0, handle};
  };

  ctx.finalize_as_graph();

  cudaGraphExec_t graphExec = NULL;
  cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  cuda_safe_call(cudaGraphLaunch(graphExec, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));
  cuda_safe_call(cudaGraphDebugDotPrint(graph, "test-while.dot", cudaGraphDebugDotFlags(0)));
#endif // !_CCCL_CTK_BELOW(12, 4)
}
