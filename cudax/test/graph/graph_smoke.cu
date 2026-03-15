//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>

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
  cuda::mr::legacy_managed_memory_resource mr{};
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

  SECTION("path_builder replicate and join")
  {
    cudax::graph_builder g;
    auto pb = cudax::start_path(g);
    cudax::launch(pb, test::one_thread_dims, test::assign_42{}, ptr);

    auto branches = cudax::replicate<2>(pb);
    CUDAX_REQUIRE(branches[0].get_dependencies().size() == 0);
    CUDAX_REQUIRE(branches[1].get_dependencies().size() == 0);

    cudax::join(branches, pb);
    CUDAX_REQUIRE(branches[0].get_dependencies().size() == 1);
    CUDAX_REQUIRE(branches[1].get_dependencies().size() == 1);
    CUDAX_REQUIRE(branches[0].get_dependencies()[0] == pb.get_dependencies()[0]);
    CUDAX_REQUIRE(branches[1].get_dependencies()[0] == pb.get_dependencies()[0]);

    cudax::launch(branches[0], test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(branches[1], test::one_thread_dims, test::atomic_add_one{}, ptr);

    auto sink = cudax::start_path(g);
    cudax::join(sink, branches);
    cudax::launch(sink, test::one_thread_dims, test::verify_n<44>{}, ptr);

    auto exec = g.instantiate();
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 44);
    *ptr = 0;
  }

  SECTION("path_builder join supports group-to-group")
  {
    cudax::graph_builder g;
    auto root = cudax::start_path(g);
    cudax::launch(root, test::one_thread_dims, test::assign_42{}, ptr);

    auto source0 = cudax::start_path(g, root);
    auto source1 = cudax::start_path(g, root);
    cudax::launch(source0, test::one_thread_dims, test::atomic_add_one{}, ptr);
    cudax::launch(source1, test::one_thread_dims, test::atomic_add_one{}, ptr);

    auto target0 = cudax::start_path(g);
    auto target1 = cudax::start_path(g);
    auto targets = ::cuda::std::array<cudax::path_builder, 2>{target0, target1};
    auto sources = ::cuda::std::array<cudax::path_builder, 2>{source0, source1};

    cudax::join(targets, sources);
    CUDAX_REQUIRE(targets[0].get_dependencies().size() == 2);
    CUDAX_REQUIRE(targets[1].get_dependencies().size() == 2);

    cudax::launch(targets[0], test::one_thread_dims, test::verify_n<44>{}, ptr);
    auto exec = g.instantiate();
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 44);
    *ptr = 0;
  }

  SECTION("path_builder join supports multi-node source fan-in")
  {
    cudax::graph_builder g;
    auto root = cudax::start_path(g);
    cudax::launch(root, test::one_thread_dims, test::assign_42{}, ptr);

    auto p0 = cudax::start_path(g, root);
    auto p1 = cudax::start_path(g, root);
    auto p2 = cudax::start_path(g, root);
    auto p3 = cudax::start_path(g, root);
    auto n0 = cudax::launch(p0, test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto n1 = cudax::launch(p1, test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto n2 = cudax::launch(p2, test::one_thread_dims, test::atomic_add_one{}, ptr);
    auto n3 = cudax::launch(p3, test::one_thread_dims, test::atomic_add_one{}, ptr);

    auto source0 = cudax::start_path(g);
    auto source1 = cudax::start_path(g);
    source0.depends_on(n0, n1);
    source1.depends_on(n2, n3);
    CUDAX_REQUIRE(source0.get_dependencies().size() == 2);
    CUDAX_REQUIRE(source1.get_dependencies().size() == 2);

    auto target0 = cudax::start_path(g);
    auto target1 = cudax::start_path(g);
    auto targets = ::cuda::std::array<cudax::path_builder, 2>{target0, target1};
    auto sources = ::cuda::std::array<cudax::path_builder, 2>{source0, source1};
    cudax::join(targets, sources);
    CUDAX_REQUIRE(targets[0].get_dependencies().size() == 4);
    CUDAX_REQUIRE(targets[1].get_dependencies().size() == 4);

    cudax::launch(targets[0], test::one_thread_dims, test::verify_n<46>{}, ptr);
    auto exec = g.instantiate();
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(*ptr == 46);
    *ptr = 0;
  }

  SECTION("path_builder replicate_prepend variants")
  {
    cudax::graph_builder g;

    auto dynamic_seed = cudax::start_path(g);
    cudax::launch(dynamic_seed, test::one_thread_dims, test::assign_42{}, ptr);
    const auto dynamic_seed_dep = dynamic_seed.get_dependencies()[0];
    auto dynamic_group          = cudax::replicate_prepend(std::move(dynamic_seed), 2);

    CUDAX_REQUIRE(dynamic_group.size() == 3);
    CUDAX_REQUIRE(dynamic_group[0].get_dependencies().size() == 1);
    CUDAX_REQUIRE(dynamic_group[0].get_dependencies()[0] == dynamic_seed_dep);
    CUDAX_REQUIRE(dynamic_group[1].get_dependencies().size() == 0);
    CUDAX_REQUIRE(dynamic_group[2].get_dependencies().size() == 0);

    auto static_seed = cudax::start_path(g);
    cudax::launch(static_seed, test::one_thread_dims, test::assign_42{}, ptr);
    const auto static_seed_dep = static_seed.get_dependencies()[0];
    auto static_group          = cudax::replicate_prepend<2>(std::move(static_seed));

    CUDAX_REQUIRE(static_group.size() == 3);
    CUDAX_REQUIRE(static_group[0].get_dependencies().size() == 1);
    CUDAX_REQUIRE(static_group[0].get_dependencies()[0] == static_seed_dep);
    CUDAX_REQUIRE(static_group[1].get_dependencies().size() == 0);
    CUDAX_REQUIRE(static_group[2].get_dependencies().size() == 0);
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
    SECTION("Multi-device path_builder join")
    {
      cudax::graph_builder g(cuda::devices[0]);
      auto root = cudax::start_path(g);
      cudax::launch(root, test::one_thread_dims, test::assign_42{}, ptr);

      auto dev0_source = cudax::start_path(cuda::devices[0], root);
      auto dev1_source = cudax::start_path(cuda::devices[1], root);
      cudax::launch(dev0_source, test::one_thread_dims, test::atomic_add_one{}, ptr);
      cudax::launch(dev1_source, test::one_thread_dims, test::atomic_add_one{}, ptr);

      auto target_group = cudax::replicate<1>(cudax::start_path(cuda::devices[0], root));
      auto source_group = ::cuda::std::array<cudax::path_builder, 2>{dev0_source, dev1_source};
      cudax::join(target_group, source_group);
      CUDAX_REQUIRE(target_group[0].get_dependencies().size() >= 2);

      cudax::launch(target_group[0], test::one_thread_dims, test::verify_n<44>{}, ptr);
      auto exec = g.instantiate();
      exec.launch(s);
      s.sync();
      CUDAX_REQUIRE(*ptr == 44);
      *ptr = 0;
    }

    SECTION("Multi-device graph")
    {
      cuda::device_memory_pool_ref dev0_mr = cuda::device_default_memory_pool(cuda::devices[0]);
      int* dev0_ptr                        = static_cast<int*>(dev0_mr.allocate_sync(sizeof(int)));
      cuda::device_memory_pool_ref dev1_mr = cuda::device_default_memory_pool(cuda::devices[1]);
      int* dev1_ptr                        = static_cast<int*>(dev1_mr.allocate_sync(sizeof(int)));

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
