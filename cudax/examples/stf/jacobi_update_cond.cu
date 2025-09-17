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
 * @brief Jacobi method using the update_cond helper for clean condition management
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

int main(int argc, char** argv)
{
  stackable_ctx ctx;

  size_t n     = 4096;
  size_t m     = 4096;
  double tol   = 0.1;
  int max_iter = 1000;

  if (argc > 2)
  {
    n = atol(argv[1]);
    m = atol(argv[2]);
  }

  if (argc > 3)
  {
    tol = atof(argv[3]);
  }

  if (argc > 4)
  {
    max_iter = atoi(argv[4]);
  }

  auto lA        = ctx.logical_data(shape_of<slice<double, 2>>(m, n));
  auto lAnew     = ctx.logical_data(lA.shape());
  auto lresidual = ctx.logical_data(shape_of<scalar_view<double>>());
  auto liter     = ctx.logical_data(shape_of<scalar_view<int>>());

  ctx.parallel_for(lA.shape(), lA.write(), lAnew.write()).set_symbol("init")->*
    [=] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j)    = (i == j) ? 1.0 : -1.0;
      Anew(i, j) = A(i, j);
    };

  // Initialize iteration counter
  ctx.parallel_for(box(1), liter.write())->*[] __device__(size_t, auto iter) {
    *iter = 0;
  };

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));
  cuda_safe_call(cudaEventRecord(start, ctx.fence()));

  {
    auto while_guard = ctx.while_graph_scope();

    ctx.parallel_for(inner<1>(lA.shape()), lA.read(), lAnew.write(), lresidual.reduce(reducer::maxval<double>()))
        ->*[tol] __device__(size_t i, size_t j, auto A, auto Anew, auto& residual) {
              Anew(i, j)   = 0.25 * (A(i - 1, j) + A(i + 1, j) + A(i, j - 1) + A(i, j + 1));
              double error = fabs(A(i, j) - Anew(i, j));
              residual     = ::std::max(error, residual);
            };

    ctx.parallel_for(inner<1>(lA.shape()), lA.rw(), lAnew.read())->*[] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = Anew(i, j);
    };

    while_guard.update_cond(lresidual.read(), liter.rw())->*[tol, max_iter] __device__(auto residual, auto iter) {
      bool converged   = (*residual < tol);
      bool max_reached = ((*iter)++ >= max_iter); // Maximum iteration limit
      return !converged && !max_reached; // Continue if not converged and under limit
    };
  }

  cuda_safe_call(cudaEventRecord(stop, ctx.fence()));

  printf("Converged after %d iterations, residual = %lf\n", ctx.wait(liter), ctx.wait(lresidual));

  ctx.finalize();

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time: %f ms\n", elapsedTime);

  return 0;
}
