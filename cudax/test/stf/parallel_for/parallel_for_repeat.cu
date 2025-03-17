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
 * @brief This creates a dummy grid of devices (repeating device 0 multiple
 *         times) to check that parallel_for works on a grid of places.
 */

#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

int main()
{
  context ctx;

  const int N = 1024 * 1024 * 32;
  double *X, *Y;

  X = new double[N];
  Y = new double[N];

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = X0(ind);
    Y[ind] = Y0(ind);
  }

  auto handle_X = ctx.logical_data(X, {N});
  auto handle_Y = ctx.logical_data(Y, {N});

  auto where = exec_place::repeat(exec_place::current_device(), 8);

  double alpha = 3.14;
  size_t NITER = 5;

  /* Compute Y = Y + alpha X */
  for (size_t k = 0; k < NITER; k++)
  {
    ctx.parallel_for(tiled_partition<1024 * 1024>(), where, handle_X.shape(), handle_X.read(), handle_Y.rw())
        ->*[=] _CCCL_DEVICE(size_t i, auto sX, auto sY) {
              sY(i) += alpha * sX(i);
            };
  }

  ctx.finalize();

  for (size_t ind = 0; ind < N; ind++)
  {
    // Y should be Y0 + NITER alpha X0
    assert(fabs(Y[ind] - (Y0(ind) + NITER * alpha * X0(ind))) < 0.0001);

    // X should be X0
    assert(fabs(X[ind] - X0(ind)) < 0.0001);
  }
}
