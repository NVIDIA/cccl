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
 * @brief Test explicit uses of the API to change epoch and create a sequence
 *        of CUDA graphs
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

int main()
{
  graph_ctx ctx;

  const size_t N     = 8;
  const size_t NITER = 2;

  double A[N];
  for (size_t i = 0; i < N; i++)
  {
    A[i] = 1.0 * i;
  }

  auto lA = ctx.logical_data(A);

  for (size_t k = 0; k < NITER; k++)
  {
    ctx.parallel_for(blocked_partition(), exec_place::current_device(), lA.shape(), lA.rw())
        ->*[] __host__ __device__(size_t i, slice<double> A) { A(i) = cos(A(i)); };

    ctx.change_epoch();
  }

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    double Ai_ref = 1.0 * i;
    for (size_t k = 0; k < NITER; k++)
    {
      Ai_ref = cos(Ai_ref);
    }

    EXPECT(fabs(A[i] - Ai_ref) < 0.01);
  }
}
