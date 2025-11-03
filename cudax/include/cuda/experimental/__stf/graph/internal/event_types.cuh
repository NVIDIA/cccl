//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>

#include <vector>

namespace cuda::experimental::stf::reserved
{
class graph_event_impl;

using graph_event = reserved::handle<graph_event_impl, reserved::handle_flags::non_null>;

/* A prereq corresponds to a node, which is always paired to its graph */
class graph_event_impl : public event_impl
{
protected:
  graph_event_impl()                        = default;
  graph_event_impl(const graph_event_impl&) = delete;
  graph_event_impl(cudaGraphNode_t n, size_t stage, cudaGraph_t g)
      : node(n)
      , stage(stage)
      , g(g)
  {
    assert(node);
  }

  // Remove duplicate entries from a vector.
  //
  // We could use sort and unique, but that could change the order of the nodes
  // in the result, so that we would generate graphs with a topology that
  // changes across multiple calls, where the update method would fail.
  // Instead, we only append nodes to the result vector if they were not seen
  // before.
  void remove_duplicates(::std::vector<cudaGraphNode_t>& nodes)
  {
    ::std::unordered_set<cudaGraphNode_t> seen;
    ::std::vector<cudaGraphNode_t> result;

    for (cudaGraphNode_t node : nodes)
    {
      if (seen.insert(node).second)
      {
        result.push_back(node); // First time we've seen this node
      }
    }

    ::std::swap(nodes, result);
  }

  bool factorize(backend_ctx_untyped& bctx, reserved::event_vector& events) override
  {
    _CCCL_ASSERT(events.size() >= 2, "invalid value");

    // Sanity checks to ensure we are manipulating events in the CUDA graph backend
    for (const auto& e : events)
    {
      _CCCL_ASSERT(dynamic_cast<const graph_event_impl*>(e.operator->()), "invalid event type");
    }

    auto bctx_stage        = bctx.stage();
    cudaGraph_t bctx_graph = bctx.graph();

    // To prevent "infinite" growth of event lists, we factorize long vector of
    // graph events by making them depend on a single node instead
    if (events.size() > 16)
    {
      cudaGraphNode_t n;

      ::std::vector<cudaGraphNode_t> nodes;

      // List all graph nodes in the vector of events
      for (const auto& e : events)
      {
        const auto ge = dynamic_cast<const graph_event_impl*>(e.operator->());

        // the current stage cannot be smaller than existing events
        _CCCL_ASSERT(bctx_stage >= ge->stage, "");

        if (ge->stage == bctx_stage)
        {
          nodes.push_back(ge->node);

          // We can only have nodes from the same graph
          _CCCL_ASSERT(bctx_graph == ge->g, "inconsistent graphs events (different graphs)");
        }
      }

      // Note : we do nothing if the list is empty : we will just clear the
      // events. This could happen if all events where in a previous stage.
      if (nodes.size() == 1)
      {
        n = nodes[0];
      }
      else if (nodes.size() > 1)
      {
        // We cannot have duplicate entries in dependencies
        remove_duplicates(nodes);

        // Create a new empty graph node which depends on the previous ones,
        // empty the list of events and replace it with this single "empty" event
        cuda_safe_call(cudaGraphAddEmptyNode(&n, bctx_graph, nodes.data(), nodes.size()));
      }

      events.clear();

      if (nodes.size() > 0)
      {
        events.push_back(graph_event(n, bctx_stage, bctx_graph));
      }

      return true;
    }

    return false;
  }

public:
  mutable cudaGraphNode_t node;
  mutable size_t stage;
  mutable cudaGraph_t g;
};

// This converts a prereqs and converts it to a vector of graph nodes. As a
// side-effect, it also remove duplicates from the prereqs list of events
inline ::std::vector<cudaGraphNode_t>
join_with_graph_nodes(backend_ctx_untyped& bctx, event_list& prereqs, size_t current_stage)
{
  ::std::vector<cudaGraphNode_t> nodes;

  // CUDA Graph API does not want to have the same dependency passed multiple times
  prereqs.optimize(bctx);

  for (const auto& e : prereqs)
  {
    const auto ge = reserved::graph_event(e, reserved::use_dynamic_cast);
    EXPECT(current_stage >= ge->stage);

    // If current_stage > ge->stage, then this was already implicitly
    // synchronized as different stages are submitted sequentially.
    if (current_stage == ge->stage)
    {
      nodes.push_back(ge->node);
    }
  }

  return nodes;
}

// This creates a new CUDASTF event list from a cudaGraphNode_t and sets the appropriate annotations in the DOT output.
/* previous_prereqs is only passed so that we can insert the proper DOT annotations */
template <typename context_t>
inline void fork_from_graph_node(
  context_t& ctx,
  cudaGraphNode_t n,
  cudaGraph_t g,
  size_t stage,
  event_list& previous_prereqs,
  ::std::string prereq_string)
{
  auto gnp = reserved::graph_event(n, stage, g);
  gnp->set_symbol(ctx, mv(prereq_string));

  auto& dot = *ctx.get_dot();
  if (dot.is_tracing_prereqs())
  {
    for (const auto& e : previous_prereqs)
    {
      dot.add_edge(e->unique_prereq_id, gnp->unique_prereq_id, edge_type::prereqs);
    }
  }

  previous_prereqs = event_list(gnp);
}
} // namespace cuda::experimental::stf::reserved
