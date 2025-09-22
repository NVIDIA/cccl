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

/**
 * @brief Insert an existing CUDA graph node into a graph context with appropriate dependencies
 *
 * This function is designed for graph contexts and adds the provided graph node
 * to the context's graph with dependencies from the input prerequisites.
 *
 * @tparam ctx_t Context type (must be a graph_ctx or context using graph_ctx under the hood)
 * @param ctx The execution context (must be a graph context)
 * @param node The existing CUDA graph node to insert
 * @param input_prereqs Input dependencies that must be satisfied (must be graph events)
 * @return event_list Events representing the completion of the graph node insertion
 */
template <typename ctx_t>
event_list insert_graph_node(ctx_t& ctx, cudaGraphNode_t node, event_list& input_prereqs)
{
  // This function is specifically designed for graph contexts
  _CCCL_ASSERT(ctx.is_graph_ctx(), "insert_graph_node can only be used with graph contexts");

  cudaGraph_t support_graph = ctx.graph();
  size_t graph_stage        = ctx.stage();

  // Insert assertions that the input_prereqs events are graph events
  // that can be used in the support_graph
#  ifndef NDEBUG
  for (const auto& e : input_prereqs)
  {
    const auto ge = graph_event(e, use_dynamic_cast);
    _CCCL_ASSERT(ge, "Expected graph event for graph context");
    _CCCL_ASSERT(ge->g == support_graph, "Graph event must belong to the same graph");
  }
#  endif

  ::std::vector<cudaGraphNode_t> ready_nodes = join_with_graph_nodes(ctx, input_prereqs, graph_stage);

  // Add dependencies from the ready_nodes to the existing node
  if (!ready_nodes.empty())
  {
#  if _CCCL_CTK_AT_LEAST(13, 0)
    cuda_safe_call(cudaGraphAddDependencies(support_graph, ready_nodes.data(), &node, nullptr, ready_nodes.size()));
#  else // _CCCL_CTK_AT_LEAST(13, 0)
    cuda_safe_call(cudaGraphAddDependencies(support_graph, ready_nodes.data(), &node, ready_nodes.size()));
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  }

  // Create an event that depends on the inserted graph node
  auto node_event = graph_event(node, graph_stage, support_graph);
  node_event->set_symbol(ctx, "inserted_graph_node");

  // Return the event list from that single event
  return event_list(mv(node_event));
}

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

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, ctx.graph(), 1, cudaGraphCondAssignDefault);

  cudaGraphNodeParams cParams = {};
  cParams.type                = cudaGraphNodeTypeConditional;
  cParams.conditional.handle  = handle;
  cParams.conditional.type    = cudaGraphCondTypeWhile;
  cParams.conditional.size    = 1;

  cudaGraphNode_t conditionalNode;
  // There is no input dependencies yet, we will add them later
#  if _CCCL_CTK_AT_LEAST(13, 0)
  cudaGraphAddNode(&conditionalNode, ctx.graph(), nullptr, nullptr, 0, &cParams);
#  else
  cudaGraphAddNode(&conditionalNode, ctx.graph(), nullptr, 0, &cParams);
#  endif

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  graph_ctx sub_ctx(bodyGraph);

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

  event_list cond_graph_launched = reserved::insert_graph_node(ctx, conditionalNode, fX_get_events);

  fX.unfreeze(cond_graph_launched);

  ctx.host_launch(lX.read())->*[](auto x) {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
      EXPECT(x(i) == 3 * X0(i) + 2 * 5);
    }
  };

  ctx.finalize();
#endif // !_CCCL_CTK_BELOW(12, 4)
}
