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
 * @brief Graph utility functions for CUDA graph manipulation and integration
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/graph/internal/event_types.cuh>
#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

namespace cuda::experimental::stf::reserved
{

/**
 * @brief Template function to launch a graph executable
 *
 * @tparam ctx_t Context type (stream_ctx, graph_ctx, or unified context)
 * @param ctx The execution context
 * @param graph_exec The executable graph to launch
 * @param input_prereqs Input dependencies that must be satisfied before launch
 * @return event_list Events representing the completion of the graph launch
 */
template <typename ctx_t>
event_list graph_exec_launch(ctx_t& ctx, cudaGraphExec_t graph_exec, event_list& input_prereqs)
{
  auto support_dstream = ctx.pick_dstream();

  // The graph launch depends on the input events, the resulting events will be implied by the stream semantic so we can
  // ignore them here
  /* auto before_launch = */ join_with_stream(ctx, support_dstream, input_prereqs, "graph_launch", false);

  cuda_safe_call(cudaGraphLaunch(graph_exec, support_dstream.stream));

  event_list graph_launched;
  graph_launched.sync_with_stream(ctx, support_dstream.stream);

  return graph_launched;
}

/**
 * @brief Insert a CUDA graph into a context with appropriate backend handling
 *
 * For graph contexts: Inserts the graph as a child graph node
 * For stream contexts: Instantiates and launches the graph
 *
 * @tparam ctx_t Context type (stream_ctx, graph_ctx, or unified context)
 * @param ctx The execution context
 * @param graph The CUDA graph to insert
 * @param input_prereqs Input dependencies that must be satisfied
 * @return event_list Events representing the completion of the graph operation
 */
template <typename ctx_t>
event_list insert_graph(ctx_t& ctx, cudaGraph_t graph, event_list& input_prereqs)
{
  // If this is a graph context, we will insert this graph as a child graph,
  // otherwise we instantiate it and launch it.
  if (ctx.is_graph_ctx())
  {
    cudaGraph_t support_graph = ctx.graph();
    size_t graph_stage        = ctx.stage();

    // Insert assertions that the input_prereqs events are graph events
    // that can be used in the support_graph
#ifndef NDEBUG
    for (const auto& e : input_prereqs)
    {
      const auto ge = graph_event(e, use_dynamic_cast);
      _CCCL_ASSERT(ge, "Expected graph event for graph context");
      _CCCL_ASSERT(ge->g == support_graph, "Graph event must belong to the same graph");
    }
#endif

    ::std::vector<cudaGraphNode_t> ready_nodes = join_with_graph_nodes(ctx, input_prereqs, graph_stage);

    // Create a child node from the graph that depends on ready_nodes and add it to support_graph
    cudaGraphNode_t child_graph_node;
    cuda_safe_call(
      cudaGraphAddChildGraphNode(&child_graph_node, support_graph, ready_nodes.data(), ready_nodes.size(), graph));

    // Create an event that depends on the child graph node (convert it to an event itself)
    auto child_event = graph_event(child_graph_node, graph_stage, support_graph);
    child_event->set_symbol(ctx, "inserted_graph");

    // Return the event list from that single event
    return event_list(mv(child_event));
  }

  cudaGraphExec_t graph_exec = NULL;
  cuda_safe_call(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

  return graph_exec_launch(ctx, graph_exec, input_prereqs);
}

} // namespace cuda::experimental::stf::reserved
