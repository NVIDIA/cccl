//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/event.cuh>
#include <cuda/experimental/graph.cuh>
#include <cuda/experimental/stream.cuh>

#include <testing.cuh>
#include <utility.cuh>

namespace
{

// Empty node descriptor for testing
struct empty_node_descriptor
{
  cuda::experimental::graph_node_ref __add_to_graph(cudaGraph_t graph, _CUDA_VSTD::span<cudaGraphNode_t> deps) const
  {
    cudaGraphNode_t node;
    _CCCL_TRY_CUDA_API(cudaGraphAddEmptyNode, "cudaGraphAddEmptyNode failed", &node, graph, deps.data(), deps.size());
    return cuda::experimental::graph_node_ref{node, graph};
  }
};

} // namespace

C2H_TEST("can default construct a graph and destroy it", "[graph]")
{
  cuda::experimental::graph_builder g;
  CUDAX_REQUIRE(g.get() != nullptr);
}

C2H_TEST("can create an empty node in a graph", "[graph]")
{
  cuda::experimental::graph_builder g;
  auto node = g.add(empty_node_descriptor{});
  CUDAX_REQUIRE(node.get() != nullptr);
  CUDAX_REQUIRE(node.type() == cuda::experimental::graph_node_type::empty);
}

C2H_TEST("can create multiple nodes and establish dependencies", "[graph]")
{
  cuda::experimental::graph_builder g;

  // Create three empty nodes
  auto node1 = g.add(empty_node_descriptor{});
  auto node2 = g.add(empty_node_descriptor{});
  auto node3 = g.add(empty_node_descriptor{});

  // Set up dependencies: node3 depends on node1 and node2
  node3.depends_on(node1, node2);

  // Verify the nodes exist
  CUDAX_REQUIRE(node1.get() != nullptr);
  CUDAX_REQUIRE(node2.get() != nullptr);
  CUDAX_REQUIRE(node3.get() != nullptr);

  // Verify node types
  CUDAX_REQUIRE(node1.type() == cuda::experimental::graph_node_type::empty);
  CUDAX_REQUIRE(node2.type() == cuda::experimental::graph_node_type::empty);
  CUDAX_REQUIRE(node3.type() == cuda::experimental::graph_node_type::empty);
}

C2H_TEST("can instantiate and launch a graph", "[graph]")
{
  cuda::experimental::graph_builder g;

  // Create a simple graph with two nodes
  auto node1 = g.add(empty_node_descriptor{});
  auto node2 = g.add(empty_node_descriptor{});
  node2.depends_on(node1);

  // Instantiate the graph
  auto exec = g.instantiate();
  CUDAX_REQUIRE(exec.get() != nullptr);

  // Create a stream and launch the graph
  cuda::experimental::stream s;
  exec.launch(s);

  // Wait for completion
  s.sync();
}

C2H_TEST("graph_node_ref comparison operators work correctly", "[graph]")
{
  cuda::experimental::graph_builder g;

  // Create two nodes
  auto node1 = g.add(empty_node_descriptor{});
  auto node2 = g.add(empty_node_descriptor{}, cuda::experimental::depends_on(node1.get()));

  // Test equality operators
  CUDAX_REQUIRE(node1 == node1);
  CUDAX_REQUIRE(node1 != node2);
  REQUIRE_FALSE(node1 == node2);
  REQUIRE_FALSE(node1 != node1);

  // Test null comparison
  cuda::experimental::graph_node_ref null_ref;
  REQUIRE_FALSE(node1 == null_ref);
  CUDAX_REQUIRE(node1 != null_ref);
  REQUIRE_FALSE(null_ref == node1);
  CUDAX_REQUIRE(null_ref != node1);
}

C2H_TEST("graph_node_ref can be swapped", "[graph]")
{
  cuda::experimental::graph_builder g;

  // Create two nodes
  auto node1 = g.add(empty_node_descriptor{});
  auto node2 = g.add(empty_node_descriptor{}, cuda::experimental::depends_on(node1));

  // Store original handles
  auto node1_handle = node1.get();
  auto node2_handle = node2.get();

  // Swap the nodes
  node1.swap(node2);

  // Verify the handles were swapped
  CUDAX_REQUIRE(node1.get() == node2_handle);
  CUDAX_REQUIRE(node2.get() == node1_handle);
}

C2H_TEST("graph_node_ref can be copied", "[graph]")
{
  cuda::experimental::graph_builder g;

  // Create a node
  auto node1        = g.add(empty_node_descriptor{});
  auto node1_handle = node1.get();

  // Move construct a new node
  auto node2 = node1;

  // Verify the handle was moved
  CUDAX_REQUIRE(node2.get() == node1_handle);
  CUDAX_REQUIRE(node1.get() == node1_handle);

  // Test move assignment
  auto node3 = g.add(empty_node_descriptor{});
  node3      = std::move(node2);

  // Verify the source node is still valid (moving a node ref does not zero out the source)
  CUDAX_REQUIRE(node3.get() == node1_handle);
  CUDAX_REQUIRE(node2.get() == node1_handle);
}
