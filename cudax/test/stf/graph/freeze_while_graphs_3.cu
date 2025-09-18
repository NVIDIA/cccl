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

#include <cuda/experimental/__stf/utility/graph_utilities.cuh>
#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)
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
