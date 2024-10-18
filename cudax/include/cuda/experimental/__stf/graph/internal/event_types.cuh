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

#include <vector>

namespace cuda::experimental::stf::reserved
{

/* A prereq corresponds to a node, which is always paired to its graph */
class graph_event_impl : public event_impl
{
protected:
  graph_event_impl()                        = default;
  graph_event_impl(const graph_event_impl&) = delete;
  graph_event_impl(cudaGraphNode_t n, size_t epoch)
      : node(n)
      , epoch(epoch)
  {
    assert(node);
  }

public:
  mutable cudaGraphNode_t node;
  mutable size_t epoch;
};

using graph_event = reserved::handle<graph_event_impl, reserved::handle_flags::non_null>;

// This converts a prereqs and converts it to a vector of graph nodes. As a
// side-effect, it also remove duplicates from the prereqs list of events
inline ::std::vector<cudaGraphNode_t> join_with_graph_nodes(event_list& prereqs, size_t current_epoch)
{
  ::std::vector<cudaGraphNode_t> nodes;

  // CUDA Graph API does not want to have the same dependency passed multiple times
  prereqs.optimize();

  for (const auto& e : prereqs)
  {
    const auto ge = reserved::graph_event(e, reserved::use_dynamic_cast);
    EXPECT(current_epoch >= ge->epoch);

    // If current_epoch > ge->epoch, then this was already implicitely
    // synchronized as different epochs are submitted sequentially.
    if (current_epoch == ge->epoch)
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
  context_t& ctx, cudaGraphNode_t n, size_t epoch, event_list& previous_prereqs, ::std::string prereq_string)
{
  auto gnp = reserved::graph_event(n, epoch);
  gnp->set_symbol(ctx, mv(prereq_string));

  auto& dot = *ctx.get_dot();
  if (dot.is_tracing_prereqs())
  {
    for (const auto& e : previous_prereqs)
    {
      dot.add_edge(e->unique_prereq_id, gnp->unique_prereq_id, 1);
    }
  }

  previous_prereqs = event_list(gnp);
}

} // namespace cuda::experimental::stf::reserved
