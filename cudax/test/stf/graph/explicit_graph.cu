//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! @file
//! @brief Add tasks to a user-provided graph

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

int main()
{
  cudaGraph_t graph;
  cudaGraphExec_t graphExec = NULL;
  cudaStream_t stream;

  cuda_safe_call(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cuda_safe_call(cudaGraphCreate(&graph, 0));

  graph_ctx ctx(graph);

  auto lX = ctx.token();
  auto lY = ctx.token();
  auto lZ = ctx.token();

  ctx.task(lX.write())->*[](cudaStream_t s) {
    dummy<<<1, 1, 0, s>>>();
  };

  ctx.task(lX.read(), lY.write())->*[](cudaStream_t s) {
    dummy<<<1, 1, 0, s>>>();
  };

  ctx.task(lX.read(), lZ.write())->*[](cudaStream_t s) {
    dummy<<<1, 1, 0, s>>>();
  };

  ctx.task(lY.rw(), lZ.rw())->*[](cudaStream_t s) {
    dummy<<<1, 1, 0, s>>>();
  };

  ctx.finalize_as_graph();

  cuda_safe_call(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  cuda_safe_call(cudaGraphLaunch(graphExec, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));
}
