//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//!
//! \brief Freeze a logical data in a graph to use it in the body of a "while" graph node

#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)
int X0(int i)
{
  return 17 * i + 45;
}

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
  const int N = 16;
  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  graph_ctx ctx;

  auto lX = ctx.logical_data(X);

  ctx.parallel_for(lX.shape(), lX.rw())->*[] __device__(size_t i, auto x) {
    x(i) *= 3;
  };

  auto fX = ctx.freeze(lX, access_mode::rw, data_place::current_device());

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, ctx.get_graph(), 1, cudaGraphCondAssignDefault);

  // Create a graph that will later be inserted as a child graph once all input
  // dependencies are known
  cudaGraph_t sub_graph;
  cuda_safe_call(cudaGraphCreate(&sub_graph, 0));

  // Create a context based on this child graph which is the body of the
  graph_ctx sub_ctx(sub_graph);

  auto [frozen_X, fX_get_events] = fX.get(data_place::current_device());

  auto lX_alias = sub_ctx.logical_data(frozen_X, data_place::current_device());

  sub_ctx.parallel_for(lX.shape(), lX_alias.rw())->*[] __device__(size_t i, auto x) {
    x(i) = x(i) + 2;
  };

  // We want to repeat this a fixed number of times
  sub_ctx.cuda_kernel()->*[handle]() {
    return cuda_kernel_desc{setHandle, 1, 1, 0, handle};
  };

  sub_ctx.finalize_as_graph();

  // We now create a conditional graph which depends on the same dependencies
  // as the inner ctx. We then insert the body of the graph as a child graph of
  // the conditional node because we cannot decide what graph is the body of
  // the conditional node ourselves, and we cannot add input dependencies to
  // the conditional node after it was added.

  // The child graph depends on the events to get the frozen data
  ::std::vector<cudaGraphNode_t> fX_ready_nodes = reserved::join_with_graph_nodes(ctx, fX_get_events, ctx.stage());

  cudaGraphNodeParams cParams = {};
  cParams.type                = cudaGraphNodeTypeConditional;
  cParams.conditional.handle  = handle;
  cParams.conditional.type    = cudaGraphCondTypeWhile;
  cParams.conditional.size    = 1;

  cudaGraphNode_t conditionalNode;
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cudaGraphAddNode(&conditionalNode, ctx.get_graph(), fX_ready_nodes.data(), nullptr, fX_ready_nodes.size(), &cParams);
#  else
  cudaGraphAddNode(&conditionalNode, ctx.get_graph(), fX_ready_nodes.data(), fX_ready_nodes.size(), &cParams);
#  endif

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  // A child graph contains the entire body
  cudaGraphNode_t child_graph_node;
  cuda_safe_call(cudaGraphAddChildGraphNode(&child_graph_node, bodyGraph, nullptr, 0, sub_ctx.get_graph()));

  // Create an event that depends on the conditional node, so that we unfreeze
  // after the completion of the while loop
  event_list child_graph_event;
  reserved::fork_from_graph_node(
    ctx, conditionalNode, ctx.get_graph(), ctx.stage(), child_graph_event, "child graph done");
  fX.unfreeze(child_graph_event);

  ctx.host_launch(lX.read())->*[](auto x) {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
      EXPECT(x(i) == 3 * X0(i) + 2 * 5);
    }
  };

  ctx.finalize();
#endif // !_CCCL_CTK_BELOW(12, 4)
}
