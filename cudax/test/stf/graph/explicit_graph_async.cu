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
//! @brief Add tasks to a user-provided graph and launch on a user-provided stream.
//!        Exercises graph_ctx(cudaGraph_t, cudaStream_t): finalize() submits the
//!        graph on the given stream and does not block; the caller synchronizes.

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

int main()
{
  cudaGraph_t graph;
  cudaStream_t stream;

  cuda_safe_call(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cuda_safe_call(cudaGraphCreate(&graph, 0));

  graph_ctx ctx(graph, stream);

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

  // Non-blocking: submits the graph on the user-provided stream
  ctx.finalize();

  cuda_safe_call(cudaStreamSynchronize(stream));
  cuda_safe_call(cudaStreamDestroy(stream));
}
