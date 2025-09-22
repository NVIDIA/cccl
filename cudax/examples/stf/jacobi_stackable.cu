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
 *
 * @brief Jacobi method with parallel_for and graphs
 *
 */

#include <cuda/experimental/stf.cuh>

#include <iostream>

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"

using namespace cuda::experimental::stf;

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  size_t n   = 4096;
  size_t m   = 4096;
  double tol = 0.1;

  if (argc > 2)
  {
    n = atol(argv[1]);
    m = atol(argv[2]);
  }

  if (argc > 3)
  {
    tol = atof(argv[3]);
  }

  auto lA    = ctx.logical_data(shape_of<slice<double, 2>>(m, n));
  auto lAnew = ctx.logical_data(lA.shape());

  ctx.parallel_for(lA.shape(), lA.write(), lAnew.write()).set_symbol("init")->*
    [=] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = (i == j) ? 1.0 : -1.0;
    };

  cudaEvent_t start, stop;

  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  cuda_safe_call(cudaEventRecord(start, ctx.fence()));

  auto lconverged = ctx.logical_data(shape_of<scalar_view<bool>>());

  size_t iter = 0;

  // Creating a conditional handle but not using it in a conditional node can make the graph instantiation fail.
  cudaGraphConditionalHandle handle;
  ctx.push_while(&handle, 1, cudaGraphCondAssignDefault);

  ctx.parallel_for(inner<1>(lA.shape()), lA.read(), lAnew.write(), lconverged.reduce(reducer::logical_and<bool>{}))
      ->*[tol] __device__(size_t i, size_t j, auto A, auto Anew, auto& converged) {
            Anew(i, j)   = 0.25 * (A(i - 1, j) + A(i + 1, j) + A(i, j - 1) + A(i, j + 1));
            double error = fabs(A(i, j) - Anew(i, j));
            converged    = converged && (error < tol);
          };

  ctx.parallel_for(inner<1>(lA.shape()), lA.rw(), lAnew.read())->*[] __device__(size_t i, size_t j, auto A, auto Anew) {
    A(i, j) = Anew(i, j);
  };

  ctx.parallel_for(box(1), lconverged.read())->*[handle] __device__(size_t, auto converged) {
    cudaGraphSetConditional(handle, !*converged);
  };

  ctx.pop();

  fprintf(stderr, "ITER %zu: converged\n", iter++);

  cuda_safe_call(cudaEventRecord(stop, ctx.fence()));

  ctx.finalize();

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time: %f ms\n", elapsedTime);
#endif
}
