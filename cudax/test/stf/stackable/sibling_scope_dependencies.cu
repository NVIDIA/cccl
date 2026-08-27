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

#include <unordered_set>
#include <vector>

using namespace cuda::experimental::stf;

static ::std::unordered_set<cudaGraphNode_t> transitive_dependencies(cudaGraphNode_t root)
{
  ::std::unordered_set<cudaGraphNode_t> visited;
  ::std::vector<cudaGraphNode_t> pending{root};

  while (!pending.empty())
  {
    cudaGraphNode_t node = pending.back();
    pending.pop_back();

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
    for (cudaGraphNode_t dependency : dependencies)
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

static ::std::vector<cudaGraphNode_t> nodes_of_type(cudaGraph_t graph, cudaGraphNodeType type)
{
  size_t count = 0;
  cuda_safe_call(cudaGraphGetNodes(graph, nullptr, &count));
  ::std::vector<cudaGraphNode_t> nodes(count);
  cuda_safe_call(cudaGraphGetNodes(graph, nodes.data(), &count));

  ::std::vector<cudaGraphNode_t> result;
  for (cudaGraphNode_t node : nodes)
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

static void expect_independent(cudaGraph_t graph, cudaGraphNodeType type)
{
  auto siblings = nodes_of_type(graph, type);
  EXPECT(siblings.size() == 2);
  EXPECT(transitive_dependencies(siblings[0]).count(siblings[1]) == 0);
  EXPECT(transitive_dependencies(siblings[1]).count(siblings[0]) == 0);
}

static void expect_ordered(cudaGraph_t graph, cudaGraphNodeType type)
{
  auto siblings = nodes_of_type(graph, type);
  EXPECT(siblings.size() == 2);
  const bool first_before_second = transitive_dependencies(siblings[1]).count(siblings[0]) != 0;
  const bool second_before_first = transitive_dependencies(siblings[0]).count(siblings[1]) != 0;
  EXPECT(first_before_second != second_before_first);
}

static void test_graph_scopes()
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

    expect_independent(outer.graph(), cudaGraphNodeTypeGraph);
    outer.launch();
  }

  ctx.finalize();
  EXPECT(a[0] == input[0]);
  EXPECT(b[0] == input[0]);
}

static void test_shared_data_graph_scopes()
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

    expect_ordered(outer.graph(), cudaGraphNodeTypeGraph);
    outer.launch();
  }

  ctx.finalize();
  EXPECT(value[0] == 2);
}

static void test_while_graph_scopes()
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

    expect_independent(outer.graph(), cudaGraphNodeTypeConditional);
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
  test_shared_data_graph_scopes();
  test_while_graph_scopes();
}
