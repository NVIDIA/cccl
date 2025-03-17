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
 * @brief Ensure we can use multiple graph contexts simultaneously and launch them
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

int main(int argc, char** argv)
{
  graph_ctx ctx;
  graph_ctx ctx_2;

  double X[1024], Y[1024];
  auto handle_X = ctx.logical_data(make_slice(X, 1024));
  auto handle_Y = ctx.logical_data(make_slice(Y, 1024));

  double Z[1024];
  auto handle_Z = ctx_2.logical_data(make_slice(Z, 1024));

  for (int k = 0; k < 10; k++)
  {
    ctx.task(handle_X.rw())->*[&](cudaStream_t s, auto /*unused*/) {
      dummy<<<1, 1, 0, s>>>();
    };
  }

  ctx.task(handle_X.read(), handle_Y.rw())->*[&](cudaStream_t s, auto /*unused*/, auto /*unused*/) {
    dummy<<<1, 1, 0, s>>>();
  };

  ctx_2.task(handle_Z.rw())->*[&](cudaStream_t s, auto /*unused*/) {
    dummy<<<1, 1, 0, s>>>();
  };

  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  ctx.submit(stream);
  ctx_2.submit(stream);

  if (argc > 1)
  {
    std::cout << "Generating DOT output in " << argv[1] << std::endl;
    ctx.print_to_dot(argv[1]);
  }

  if (argc > 2)
  {
    std::cout << "Generating DOT output in " << argv[2] << std::endl;
    ctx_2.print_to_dot(argv[2]);
  }

  ctx.finalize();
  ctx_2.finalize();
}
