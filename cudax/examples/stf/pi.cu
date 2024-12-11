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
 * @brief Approximate pi using Monte Carlo method
 *
 */

#include <cuda/experimental/stf.cuh>

#include <curand_kernel.h>
#include <stdio.h>

using namespace cuda::experimental::stf;

int main(int, char**)
{
  context ctx;
  auto lsum = ctx.logical_data(shape_of<scalar_view<size_t>>());

  size_t N = 1000000;

  ctx.parallel_for(box(N), lsum.reduce(reducer::sum<size_t>{}))->*[] __device__(size_t i, auto& sum) {
    curandState local_state;
    curand_init(1234, i, 0, &local_state);
    double x = curand_uniform_double(&local_state); // Random x in [0, 1)
    double y = curand_uniform_double(&local_state); // Random y in [0, 1)
    // Count (x,y) coordinates which are within the unit circle
    if (x * x + y * y <= 1.0)
    {
      sum++;
    }
  };

  // We get the ratio of "shots" within the unit circle and the total number of
  // "shots". The surface of the quarter of unit circle [0, 1) x [0, 1) is pi/4
  auto res      = ctx.wait(lsum);
  double pi_val = (4.0 * res) / N;

  ctx.finalize();

  _CCCL_ASSERT(fabs(pi_val - 3.1415) < 0.1, "Invalid result");
}
