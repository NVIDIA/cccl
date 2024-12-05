//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Ensure a graph context can be defined as a static variable
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

// Static graph ctx
graph_ctx ctx;

__global__ void dummy() {}

int main(int argc, char** argv)
{
  double X[1024], Y[1024];
  auto handle_X = ctx.logical_data(X);
  auto handle_Y = ctx.logical_data(Y);

  for (int k = 0; k < 10; k++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto /*unused*/) {
      dummy<<<1, 1, 0, s>>>();
    };
  }

  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t s, auto /*unused*/, auto /*unused*/) {
    dummy<<<1, 1, 0, s>>>();
  };

  ctx.submit();

  if (argc > 1)
  {
    std::cout << "Generating DOT output in " << argv[1] << std::endl;
    ctx.print_to_dot(argv[1]);
  }

  ctx.finalize();
}
