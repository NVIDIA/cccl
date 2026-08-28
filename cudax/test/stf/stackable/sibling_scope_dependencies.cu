//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace cuda::experimental::stf;

// Dependency structure of one CUDA graph, captured once so that repeated
// reachability queries do not re-enumerate edges through the CUDA API.
class graph_topology
{
public:
  explicit graph_topology(cudaGraph_t graph)
      : graph_(graph)
  {
    size_t count = 0;
    cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &count));
    nodes_.resize(count);
    cuda_safe_call(cudaGraphGetNodes(graph, nodes_.data(), &count));

    for (cudaGraphNode_t node : nodes_)
    {
      direct_deps_[node] = direct_dependencies(node);
    }
  }

  ::std::vector<cudaGraphNode_t> nodes_of_type(cudaGraphNodeType type) const
  {
    ::std::vector<cudaGraphNode_t> result;
    for (cudaGraphNode_t node : nodes_)
    {
      cudaGraphNodeType node_type;
      cuda_safe_call(cudaGraphNodeGetType(node, &node_type));
      if (node_type == type)
      {
        result.push_back(node);
      }
    }
    return result;
  }

  ::std::unordered_set<cudaGraphNode_t> transitive_dependencies(cudaGraphNode_t root) const
  {
    ::std::unordered_set<cudaGraphNode_t> visited;
    ::std::vector<cudaGraphNode_t> pending{root};

    while (!pending.empty())
    {
      cudaGraphNode_t node = pending.back();
      pending.pop_back();

      const auto it = direct_deps_.find(node);
      if (it == direct_deps_.end())
      {
        continue;
      }
      for (cudaGraphNode_t dependency : it->second)
      {
        if (visited.insert(dependency).second)
        {
          pending.push_back(dependency);
        }
      }
    }

    visited.erase(root);
    return visited;
  }

  // On failure, dump a DOT of the graph for diagnosis. Gated behind the
  // usual STF debug variables so a plain run (CI included) creates no files;
  // the failure message names the switch instead.
  void dump_dot(const char* label) const
  {
    const bool enabled =
      (getenv("CUDASTF_DUMP_GRAPHS") != nullptr) || (getenv("CUDASTF_DEBUG_STACKABLE_DOT") != nullptr);
    if (!enabled)
    {
      fprintf(stderr, "rerun with CUDASTF_DUMP_GRAPHS=1 to dump the '%s' graph as DOT\n", label);
      return;
    }
    ::std::string filename = ::std::string("sibling_scope_dependencies-") + label + ".dot";
    cuda_safe_call(cudaGraphDebugDotPrint(graph_, filename.c_str(), cudaGraphDebugDotFlagsVerbose));
    fprintf(stderr, "'%s' graph dumped to %s\n", label, filename.c_str());
  }

private:
  static ::std::vector<cudaGraphNode_t> direct_dependencies(cudaGraphNode_t node)
  {
    size_t count = 0;
#if _CCCL_CTK_AT_LEAST(13, 0)
    cuda_safe_call(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &count));
#else
    cuda_safe_call(cudaGraphNodeGetDependencies(node, nullptr, &count));
#endif
    ::std::vector<cudaGraphNode_t> dependencies(count);
#if _CCCL_CTK_AT_LEAST(13, 0)
    ::std::vector<cudaGraphEdgeData> edge_data(count);
    cuda_safe_call(cudaGraphNodeGetDependencies(node, dependencies.data(), edge_data.data(), &count));
#else
    cuda_safe_call(cudaGraphNodeGetDependencies(node, dependencies.data(), &count));
#endif
    return dependencies;
  }

  cudaGraph_t graph_;
  ::std::vector<cudaGraphNode_t> nodes_;
  ::std::unordered_map<cudaGraphNode_t, ::std::vector<cudaGraphNode_t>> direct_deps_;
};

// `what` names the failed predicate: all checks funnel through the same
// EXPECT line, so the exception message alone cannot identify which one fired.
void check_or_dump(const graph_topology& topology, bool condition, const char* label, const char* what)
{
  if (!condition)
  {
    fprintf(stderr, "graph topology check '%s' failed: %s\n", label, what);
    topology.dump_dot(label);
  }
  EXPECT(condition);
}

// Number of kernel nodes in a graph, descending into child graphs. Nested
// scopes and conditional bodies wrap their work in child graph nodes, so a
// flat count would miss it.
size_t count_kernel_nodes_recursive(cudaGraph_t graph)
{
  const graph_topology topology(graph);
  size_t count = topology.nodes_of_type(cudaGraphNodeTypeKernel).size();
  for (cudaGraphNode_t child : topology.nodes_of_type(cudaGraphNodeTypeGraph))
  {
    cudaGraph_t child_graph;
    cuda_safe_call(cudaGraphChildGraphNodeGetGraph(child, &child_graph));
    count += count_kernel_nodes_recursive(child_graph);
  }
  return count;
}

// The two sibling nodes of the given type must not depend on each other.
//
// Independence alone would also hold if the sibling scopes came out empty, so
// for child graph siblings additionally require that each body contains real
// kernel work; the value checks at the end of each test then prove that both
// siblings actually consumed the shared input. A stronger "both siblings
// depend on a common upload node" check is not expressible at this level: the
// shared input is imported by the enclosing scope, whose transfer runs in the
// enclosing context's stream, so it is not a node of this graph and sibling
// nodes correctly have no in-graph ancestor.
void expect_independent(cudaGraph_t graph, cudaGraphNodeType type, const char* label)
{
  const graph_topology topology(graph);
  auto siblings = topology.nodes_of_type(type);
  check_or_dump(topology, siblings.size() == 2, label, "expected exactly two sibling nodes");

  const auto deps0 = topology.transitive_dependencies(siblings[0]);
  const auto deps1 = topology.transitive_dependencies(siblings[1]);
  check_or_dump(topology, deps0.count(siblings[1]) == 0, label, "first sibling depends on the second");
  check_or_dump(topology, deps1.count(siblings[0]) == 0, label, "second sibling depends on the first");

  if (type == cudaGraphNodeTypeGraph)
  {
    for (cudaGraphNode_t sibling : siblings)
    {
      cudaGraph_t body;
      cuda_safe_call(cudaGraphChildGraphNodeGetGraph(sibling, &body));
      check_or_dump(topology, count_kernel_nodes_recursive(body) > 0, label, "sibling scope contains no kernel");
    }
  }
}

// Exactly one of the two sibling nodes of the given type must transitively
// depend on the other.
void expect_ordered(cudaGraph_t graph, cudaGraphNodeType type, const char* label)
{
  const graph_topology topology(graph);
  auto siblings = topology.nodes_of_type(type);
  check_or_dump(topology, siblings.size() == 2, label, "expected exactly two sibling nodes");

  const bool first_before_second = topology.transitive_dependencies(siblings[1]).count(siblings[0]) != 0;
  const bool second_before_first = topology.transitive_dependencies(siblings[0]).count(siblings[1]) != 0;
  check_or_dump(
    topology, first_before_second != second_before_first, label, "siblings are not ordered in exactly one direction");
}

void test_graph_scopes()
{
  stackable_ctx ctx;
  int input[1] = {42};
  int a[1]     = {0};
  int b[1]     = {0};
  auto din     = ctx.logical_data(input);
  auto da      = ctx.logical_data(a);
  auto db      = ctx.logical_data(b);

  {
    stackable_ctx::launchable_graph_scope outer{ctx};
    din.push(access_mode::read, data_place::current_device());
    {
      auto sibling = ctx.graph_scope();
      ctx.parallel_for(box(1), din.read(), da.write())->*[] __device__(size_t, auto in, auto out) {
        out(0) = in(0);
      };
    }
    {
      auto sibling = ctx.graph_scope();
      ctx.parallel_for(box(1), din.read(), db.write())->*[] __device__(size_t, auto in, auto out) {
        out(0) = in(0);
      };
    }

    expect_independent(outer.graph(), cudaGraphNodeTypeGraph, "graph_scopes");
    outer.launch();
  }

  ctx.finalize();
  EXPECT(a[0] == input[0]);
  EXPECT(b[0] == input[0]);
}

// Same shape as test_graph_scopes, but each sibling reads the shared input
// from a doubly-nested scope whose intermediate level never touches the data.
// This exercises the multi-hop import walk in validate_access: the read-only
// mode inherited from the outer freeze must propagate through the intermediate
// scope rather than escalate to rw.
void test_nested_graph_scopes()
{
  stackable_ctx ctx;
  int input[1] = {42};
  int a[1]     = {0};
  int b[1]     = {0};
  auto din     = ctx.logical_data(input);
  auto da      = ctx.logical_data(a);
  auto db      = ctx.logical_data(b);

  {
    stackable_ctx::launchable_graph_scope outer{ctx};
    din.push(access_mode::read, data_place::current_device());
    {
      auto mid = ctx.graph_scope();
      {
        auto inner = ctx.graph_scope();
        ctx.parallel_for(box(1), din.read(), da.write())->*[] __device__(size_t, auto in, auto out) {
          out(0) = in(0);
        };
      }
    }
    {
      auto mid = ctx.graph_scope();
      {
        auto inner = ctx.graph_scope();
        ctx.parallel_for(box(1), din.read(), db.write())->*[] __device__(size_t, auto in, auto out) {
          out(0) = in(0);
        };
      }
    }

    expect_independent(outer.graph(), cudaGraphNodeTypeGraph, "nested_graph_scopes");
    outer.launch();
  }

  ctx.finalize();
  EXPECT(a[0] == input[0]);
  EXPECT(b[0] == input[0]);
}

void test_shared_data_graph_scopes()
{
  stackable_ctx ctx;
  int value[1] = {0};
  auto data    = ctx.logical_data(value);

  {
    stackable_ctx::launchable_graph_scope outer{ctx};
    {
      auto sibling = ctx.graph_scope();
      ctx.parallel_for(box(1), data.rw())->*[] __device__(size_t, auto x) {
        x(0)++;
      };
    }
    {
      auto sibling = ctx.graph_scope();
      ctx.parallel_for(box(1), data.rw())->*[] __device__(size_t, auto x) {
        x(0)++;
      };
    }

    expect_ordered(outer.graph(), cudaGraphNodeTypeGraph, "shared_data_graph_scopes");
    outer.launch();
  }

  ctx.finalize();
  EXPECT(value[0] == 2);
}

// Documents current behavior rather than desirable behavior: a root-created
// logical data that no scope pushed read-only is imported eagerly in rw mode
// by the first sibling (avoiding a re-push if a later scope writes), so two
// read-only siblings still serialize. If the read-only inheritance in
// validate_access is ever extended to root data, this should flip to
// expect_independent, deliberately.
void test_root_read_graph_scopes()
{
  stackable_ctx ctx;
  int input[1] = {42};
  int a[1]     = {0};
  int b[1]     = {0};
  auto din     = ctx.logical_data(input);
  auto da      = ctx.logical_data(a);
  auto db      = ctx.logical_data(b);

  {
    stackable_ctx::launchable_graph_scope outer{ctx};
    {
      auto sibling = ctx.graph_scope();
      ctx.parallel_for(box(1), din.read(), da.write())->*[] __device__(size_t, auto in, auto out) {
        out(0) = in(0);
      };
    }
    {
      auto sibling = ctx.graph_scope();
      ctx.parallel_for(box(1), din.read(), db.write())->*[] __device__(size_t, auto in, auto out) {
        out(0) = in(0);
      };
    }

    expect_ordered(outer.graph(), cudaGraphNodeTypeGraph, "root_read_graph_scopes");
    outer.launch();
  }

  ctx.finalize();
  EXPECT(a[0] == input[0]);
  EXPECT(b[0] == input[0]);
}

// The independence check alone would also pass if the conditional bodies came
// out empty; require that each body actually contains a kernel. The body graph
// of an existing conditional node is only reachable through
// cudaGraphNodeGetParams, which CUDA introduced in 13.2.
#if _CCCL_CTK_AT_LEAST(13, 2) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
void expect_nonempty_conditional_bodies(cudaGraph_t graph)
{
  const graph_topology topology(graph);
  for (cudaGraphNode_t node : topology.nodes_of_type(cudaGraphNodeTypeConditional))
  {
    cudaGraphNodeParams params{};
    cuda_safe_call(cudaGraphNodeGetParams(node, &params));
    EXPECT(params.type == cudaGraphNodeTypeConditional);
    EXPECT(params.conditional.size >= 1);
    EXPECT(count_kernel_nodes_recursive(params.conditional.phGraph_out[0]) > 0);
  }
}
#endif

void test_while_graph_scopes()
{
#if _CCCL_CTK_AT_LEAST(12, 4) && !defined(CUDASTF_DISABLE_CODE_GENERATION) && defined(__CUDACC__)
  stackable_ctx ctx;
  int input[1] = {42};
  int a[1]     = {0};
  int b[1]     = {0};
  auto din     = ctx.logical_data(input);
  auto da      = ctx.logical_data(a);
  auto db      = ctx.logical_data(b);

  {
    stackable_ctx::launchable_graph_scope outer{ctx};
    din.push(access_mode::read, data_place::current_device());
    {
      auto sibling = ctx.while_graph_scope();
      ctx.parallel_for(box(1), din.read(), da.write())->*[] __device__(size_t, auto in, auto out) {
        out(0) = in(0);
      };
      sibling.update_cond()->*[] __device__ {
        return false;
      };
    }
    {
      auto sibling = ctx.while_graph_scope();
      ctx.parallel_for(box(1), din.read(), db.write())->*[] __device__(size_t, auto in, auto out) {
        out(0) = in(0);
      };
      sibling.update_cond()->*[] __device__ {
        return false;
      };
    }

    expect_independent(outer.graph(), cudaGraphNodeTypeConditional, "while_graph_scopes");
#  if _CCCL_CTK_AT_LEAST(13, 2)
    expect_nonempty_conditional_bodies(outer.graph());
#  endif
    outer.launch();
  }

  ctx.finalize();
  EXPECT(a[0] == input[0]);
  EXPECT(b[0] == input[0]);
#endif
}

int main()
{
  test_graph_scopes();
  test_nested_graph_scopes();
  test_shared_data_graph_scopes();
  test_root_read_graph_scopes();
  test_while_graph_scopes();
}
