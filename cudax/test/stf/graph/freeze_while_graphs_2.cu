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
//! \brief Freeze a logical data in a graph to use it in the body of a "while" graph node, the resulting looping graph
//! will be executed within a stream context.

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

// TODO implement this for cudaGraph with automatic instantiation (with cache),
// and for child graphs too
// TODO implement for the graph_ctx backend too. Make it a virtual method ?
template <typename ctx_t>
event_list graph_exec_launch(ctx_t& ctx, cudaGraphExec_t graph_exec, event_list& input_prereqs)
{
  auto support_dstream = ctx.pick_dstream();

  // The graph launch depends on the input events, the resulting events will be implied by the stream semantic so we can
  // ignore them here
  /* auto before_launch = */ reserved::join_with_stream(ctx, support_dstream, input_prereqs, "graph_launch", false);

  cuda_safe_call(cudaGraphLaunch(graph_exec, support_dstream.stream));

  event_list graph_launched;
  graph_launched.sync_with_stream(ctx, support_dstream.stream);

  return graph_launched;
}

template <typename ctx_t>
event_list insert_graph(ctx_t& ctx, cudaGraph_t graph, event_list& input_prereqs)
{
  // If this is a graph context, we will insert this graph as a child graph,
  // otherwise we instantiate it and launch it.
  if (ctx.is_graph_ctx()) {
      cudaGraph_t support_graph = ctx.graph();
      // TODO: Implement child graph insertion logic here
  }

  cudaGraphExec_t graph_exec = NULL;
  cuda_safe_call(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

  return graph_exec_launch(ctx, graph_exec, input_prereqs);
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

  stream_ctx ctx;

  auto lX = ctx.logical_data(X);

  ctx.parallel_for(lX.shape(), lX.rw())->*[] __device__(size_t i, auto x) {
    x(i) *= 3;
  };

  /* We are going to create a local context which is a graph, and we will populate it using a graph_ctx */
  cudaGraph_t graph;
  cuda_safe_call(cudaGraphCreate(&graph, 0));

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

  // Create a graph that will later be inserted as a child graph once all input
  // dependencies are known
  cudaGraph_t sub_graph;
  cuda_safe_call(cudaGraphCreate(&sub_graph, 0));

  // Create a context based on this child graph which is the body of the
  graph_ctx sub_ctx(sub_graph);

  auto fX                        = ctx.freeze(lX, access_mode::rw, data_place::current_device());
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

  // The child graph depends on the events to get the frozen data, so the
  // launch of the graph will depend on them

  cudaGraphNodeParams cParams = {};
  cParams.type                = cudaGraphNodeTypeConditional;
  cParams.conditional.handle  = handle;
  cParams.conditional.type    = cudaGraphCondTypeWhile;
  cParams.conditional.size    = 1;

  cudaGraphNode_t conditionalNode;
  // There is no input dependency because they are implied by graph launch
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cudaGraphAddNode(&conditionalNode, graph, nullptr, nullptr, 0, &cParams);
#  else
  cudaGraphAddNode(&conditionalNode, graph, nullptr, 0, &cParams);
#  endif

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  // A child graph contains the entire body
  cudaGraphNode_t child_graph_node;
  cuda_safe_call(cudaGraphAddChildGraphNode(&child_graph_node, bodyGraph, nullptr, 0, sub_ctx.get_graph()));

  event_list graph_launched = insert_graph(ctx, graph, fX_get_events);

  fX.unfreeze(graph_launched);

  ctx.host_launch(lX.read())->*[](auto x) {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
      EXPECT(x(i) == 3 * X0(i) + 2 * 5);
    }
  };

  ctx.finalize();
#endif // !_CCCL_CTK_BELOW(12, 4)
}
