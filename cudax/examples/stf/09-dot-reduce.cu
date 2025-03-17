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
 * @brief Implementation of the DOT kernel using a reduce access mode
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  const size_t N = 16;
  double X[N], Y[N];

  double ref_res = 0.0;

  for (size_t i = 0; i < N; i++)
  {
    X[i] = cos(double(i));
    Y[i] = sin(double(i));

    // Compute the reference result of the DOT product of X and Y
    ref_res += X[i] * Y[i];
  }

  context ctx;
  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  auto lsum = ctx.logical_data(shape_of<scalar_view<double>>());

  /* Compute sum(x_i * y_i)*/
  ctx.parallel_for(lY.shape(), lX.read(), lY.read(), lsum.reduce(reducer::sum<double>{}))
      ->*[] __device__(size_t i, auto dX, auto dY, double& sum) {
            sum += dX(i) * dY(i);
          };

  double res = ctx.wait(lsum);

  ctx.finalize();

  _CCCL_ASSERT(fabs(res - ref_res) < 0.0001, "Invalid result");
}
