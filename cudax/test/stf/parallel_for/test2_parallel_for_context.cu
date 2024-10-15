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
 * @brief Check that parallel_for constructs do update slices with shapes of different dimensions
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__host__ __device__ double x0(size_t i, size_t j)
{
  return sin((double) (i - j));
}

__host__ __device__ double y0(size_t i, size_t j)
{
  return cos((double) (i + j));
}

int main()
{
  context ctx;

  const size_t N = 16;
  double X[2 * N * 2 * N];
  double Y[N * N];

  auto lx = ctx.logical_data(make_slice(&X[0], std::tuple{2 * N, 2 * N}, 2 * N));
  auto ly = ctx.logical_data(make_slice(&Y[0], std::tuple{N, N}, N));

  ctx.parallel_for(lx.shape(), lx.write())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto sx) {
    sx(i, j) = x0(i, j);
  };

  ctx.parallel_for(ly.shape(), lx.read(), ly.write())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto sx, auto sy) {
    sy(i, j) = y0(i, j);
    for (size_t ii = 0; ii < 2; ii++)
    {
      for (size_t jj = 0; jj < 2; jj++)
      {
        sy(i, j) += sx(2 * i + ii, 2 * j + jj);
      }
    }
  };

  ctx.parallel_for(exec_place::host, ly.shape(), ly.read())->*[=] __host__(size_t i, size_t j, slice<double, 2> sy) {
    double expected = y0(i, j);
    for (size_t ii = 0; ii < 2; ii++)
    {
      for (size_t jj = 0; jj < 2; jj++)
      {
        expected += x0(2 * i + ii, 2 * j + jj);
      }
    }
    if (fabs(sy(i, j) - expected) > 0.001)
    {
      printf("sy(%zu, %zu) %f expect %f\n", i, j, sy(i, j), expected);
    }
    //    assert(fabs(sy(i, j) - expected) < 0.001);
  };

  ctx.finalize();
}
