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
 * @brief Jacobi method with parallel_for
 *
 */

#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

int main(int argc, char** argv)
{
  context ctx;

  size_t n        = 4096;
  size_t m        = 4096;
  size_t iter_max = 100;
  double tol      = 0.0000001;

  if (argc > 2)
  {
    n = atol(argv[1]);
    m = atol(argv[2]);
  }

  if (argc > 3)
  {
    iter_max = atoi(argv[3]);
  }

  if (argc > 4)
  {
    tol = atof(argv[4]);
  }

  auto lA    = ctx.logical_data(shape_of<slice<double, 2>>(m, n));
  auto lAnew = ctx.logical_data(lA.shape());

  ctx.parallel_for(lA.shape(), lA.write(), lAnew.write()).set_symbol("init")->*
    [=] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = (i == j) ? 10.0 : -1.0;
    };

  cudaEvent_t start, stop;

  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  auto lresidual = ctx.logical_data(shape_of<scalar_view<double>>());

  size_t iter = 0;
  do
  {
    ctx.parallel_for(inner<1>(lA.shape()), lA.read(), lAnew.write(), lresidual.reduce(reducer::maxval<double>{}))
        ->*[] __device__(size_t i, size_t j, auto A, auto Anew, auto residual) {
              Anew(i, j) = 0.25 * (A(i - 1, j) + A(i + 1, j) + A(i, j - 1) + A(i, j + 1));
              residual   = ::std::max(residual, fabs(A(i, j) - Anew(i, j)));
            };

    ctx.parallel_for(inner<1>(lA.shape()), lA.rw(), lAnew.read())->*[] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = Anew(i, j);
    };

    iter++;

  } while (ctx.wait(lresidual) > tol && iter < iter_max);

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time: %f ms\n", elapsedTime);
}
