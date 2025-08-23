//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/graph.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <testing.cuh>
#include <utility.cuh>

namespace
{

// Empty node descriptor for testing
struct empty_node_descriptor
{
  cuda::experimental::graph_node_ref __add_to_graph(cudaGraph_t graph, ::cuda::std::span<cudaGraphNode_t> deps) const
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
  cuda::experimental::stream s{cuda::device_ref{0}};
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

C2H_TEST("Path builder with kernel nodes", "[graph]")
{
  cudax::stream s{cuda::device_ref{0}};
  cudax::legacy_managed_memory_resource mr;
  int* ptr = static_cast<int*>(mr.allocate_sync(sizeof(int)));
  *ptr     = 0;

  SECTION("simple graph with kernel node")
  {
    cudax::graph_builder g;
    cudax::path_builder pb = cudax::start_path(g);

    // Create a kernel node
    [[maybe_unused]] auto node = cudax::launch(pb, test::one_thread_dims, test::empty_kernel{});

    auto exec = g.instantiate();
    exec.launch(s);
    s.sync();
  }

  SECTION("graph with a branching path")
  {
    cudax::graph_builder g;
    cudax::path_builder pb     = cudax::start_path(g);
    [[maybe_unused]] auto node = cudax::launch(pb, test::one_thread_dims, test::assign_42{}, ptr);
    auto node2                 = cudax::launch(pb, test::one_thread_dims, test::verify_42{}, ptr);
    auto exec                  = g.instantiate();
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 42);
    *ptr = 0;

    cudax::path_builder path1 = cudax::start_path(g, node2);
    cudax::path_builder path2 = cudax::start_path(g, node2);

    for (int i = 0; i < 10; ++i)
    {
      cudax::launch(path1, test::one_thread_dims, test::atomic_add_one{}, ptr);
    }
    for (int i = 0; i < 9; ++i)
    {
      cudax::launch(path2, test::one_thread_dims, test::atomic_sub_one{}, ptr);
    }
    CUDAX_REQUIRE(path1.get_dependencies()[0] != path2.get_dependencies()[0]);

    path1.wait(path2);
    cudax::launch(path1, test::one_thread_dims, test::verify_n<43>{}, ptr);

    auto exec2 = g.instantiate();
    exec2.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 43);
  }

  SECTION("many branching paths joining")
  {
    cudax::graph_builder g;
    cudax::path_builder pb = cudax::start_path(g);
    cudax::launch(pb, test::one_thread_dims, test::assign_42{}, ptr);

    auto node  = cudax::launch(start_path(g, pb), test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto node2 = cudax::launch(start_path(g, pb), test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto node3 = cudax::launch(start_path(g, pb), test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto node4 = cudax::launch(start_path(g, pb), test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto node5 = cudax::launch(start_path(g, pb), test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto node6 = cudax::launch(start_path(g, pb), test::one_thread_dims, test::atomic_add_one{}, ptr);

    auto another_path_builder = cudax::start_path(g, pb);
    cudax::launch(another_path_builder, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(another_path_builder, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(another_path_builder, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(another_path_builder, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(another_path_builder, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(another_path_builder, test::one_thread_dims, test::atomic_add_one{}, ptr);

    auto join_path_builder = cudax::start_path(g, node, node2, node3, another_path_builder);
    join_path_builder.depends_on(node4, node5, node6);
    CUDAX_REQUIRE(join_path_builder.get_dependencies().size() == 7);

    cudax::launch(join_path_builder, test::one_thread_dims, test::verify_n<54>{}, ptr);
    CUDAX_REQUIRE(g.node_count() == 14);
    CUDAX_REQUIRE(join_path_builder.get_dependencies().size() == 1);

    auto exec = g.instantiate();
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 54);
  }

#if _CCCL_CTK_AT_LEAST(12, 3)
  SECTION("legacy stream capture")
  {
    cudax::graph_builder g;
    cudax::path_builder pb = cudax::start_path(g);
    cudax::launch(pb, test::one_thread_dims, test::assign_42{}, ptr);

    pb.legacy_stream_capture(s, [ptr](cudaStream_t stream) {
      cudax::launch(stream, test::one_thread_dims, test::verify_42{}, ptr);
      cudax::launch(stream, test::one_thread_dims, test::atomic_add_one{}, ptr);
    });
    s.sync();
    CUDAX_REQUIRE(*ptr == 0);

    cudax::launch(pb, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(pb, test::one_thread_dims, test::verify_n<44>{}, ptr);

    CUDAX_REQUIRE(g.node_count() == 5);

    auto exec = g.instantiate();
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 44);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 3)

  if (cuda::devices.size() > 1)
  {
    SECTION("Multi-device graph")
    {
      cudax::device_memory_resource dev0_mr(cuda::devices[0]);
      int* dev0_ptr = static_cast<int*>(dev0_mr.allocate_sync(sizeof(int)));
      cudax::device_memory_resource dev1_mr(cuda::devices[1]);
      int* dev1_ptr = static_cast<int*>(dev1_mr.allocate_sync(sizeof(int)));

      cudax::graph_builder g(cuda::devices[0]);
      cudax::path_builder dev0_pb = cudax::start_path(g);

      cudax::launch(dev0_pb, test::one_thread_dims, test::assign_42{}, dev0_ptr);
      cudax::launch(dev0_pb, test::one_thread_dims, test::assign_42{}, ptr);

      cudax::path_builder dev1_pb = cudax::start_path(cuda::devices[1], dev0_pb);
      cudax::launch(dev1_pb, test::one_thread_dims, test::assign_42{}, dev1_ptr);
      cudax::launch(dev1_pb, test::one_thread_dims, test::verify_42{}, ptr);
      cudax::launch(dev1_pb, test::one_thread_dims, test::atomic_add_one{}, ptr);

      cudax::path_builder back_to_dev0 = cudax::start_path(cuda::devices[0], dev1_pb);
      cudax::launch(back_to_dev0, test::one_thread_dims, test::verify_n<43>{}, ptr);
      cudax::launch(back_to_dev0, test::one_thread_dims, test::verify_42{}, dev0_ptr);

      CUDAX_REQUIRE(g.node_count() == 7);

      auto exec = g.instantiate();
      exec.launch(s);
      s.sync();
      CUDAX_REQUIRE(*ptr == 43);

      dev0_mr.deallocate_sync(dev0_ptr, sizeof(int));
      dev1_mr.deallocate_sync(dev1_ptr, sizeof(int));
    }
  }

  mr.deallocate_sync(ptr, sizeof(int));
}
