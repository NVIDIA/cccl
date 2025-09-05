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
//! \brief Freeze a logical data in a graph to use it in a child graph

#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

int X0(int i)
{
  return 17 * i + 45;
}

__global__ void dummy() {}

int main()
{
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

  auto fX = ctx.freeze(lX, access_mode::rw, data_place::current_device());

  // Create a graph that will later be inserted as a child graph once all input
  // dependencies are known
  cudaGraph_t sub_graph;
  cuda_safe_call(cudaGraphCreate(&sub_graph, 0));

  // Create a context based on this child graph
  graph_ctx sub_ctx(sub_graph);

  auto [frozen_X, fX_get_events] = fX.get(data_place::current_device());

  auto lX_alias = sub_ctx.logical_data(frozen_X, data_place::current_device());

  // XXX we need an adapter to allocate data from the upper context
  //  auto lY = sub_ctx.logical_data(lX.shape());
  //  sub_ctx.parallel_for(lX.shape(), lX_alias.read(), lY.write())->*[] __device__(size_t i, auto x, auto y) {
  //    y(i) = x(i);
  //  };

  sub_ctx.parallel_for(lX.shape(), lX_alias.rw())->*[] __device__(size_t i, auto x) {
    x(i) = x(i) + 2;
  };

  sub_ctx.finalize_as_graph();

  // The child graph depends on the events to get the frozen data
  ::std::vector<cudaGraphNode_t> fX_ready_nodes = reserved::join_with_graph_nodes(ctx, fX_get_events, ctx.stage());

  // Add the child graph as a node that depends on the frozen data being ready
  cudaGraphNode_t child_graph_node;
  cuda_safe_call(cudaGraphAddChildGraphNode(
    &child_graph_node, ctx.get_graph(), fX_ready_nodes.data(), fX_ready_nodes.size(), sub_ctx.get_graph()));

  // Create an event that signals when the child graph completes
  event_list child_graph_event;
  reserved::fork_from_graph_node(
    ctx, child_graph_node, ctx.get_graph(), ctx.stage(), child_graph_event, "child graph done");

  // Unfreeze the data after the child graph completes
  fX.unfreeze(child_graph_event);

  ctx.host_launch(lX.read())->*[](auto x) {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
      EXPECT(x(i) == 3 * X0(i) + 2);
    }
  };

  ctx.finalize();
}
