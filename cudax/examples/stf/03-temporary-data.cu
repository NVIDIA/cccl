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
 * @brief This example illustrates how we can create temporary data from shapes, and use them in tasks
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  const int n = 4096;
  int X[n];
  int Y[n];

  for (size_t i = 0; i < n; i++)
  {
    X[i] = 3 * i;
    Y[i] = 2 * i - 3;
  }

  context ctx;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  // Select an odd number
  int niter = 19;
  assert(niter % 2 == 1);

  for (int iter = 0; iter < niter; iter++)
  {
    // We here define a temporary vector with the same shape as X, for which there is no existing copy
    // This data handle has a limited scope, so that it is automatically destroyed at each iteration of the loop
    auto tmp = ctx.logical_data(lX.shape());

    ctx.task(lY.rw(), lX.rw(), tmp.write())->*[](cudaStream_t s, auto sY, auto sX, auto sTMP) {
      // We swap X and Y using TMP as temporary buffer
      // TMP = X
      cuda_safe_call(
        cudaMemcpyAsync(sTMP.data_handle(), sX.data_handle(), n * sizeof(int), cudaMemcpyDeviceToDevice, s));
      // X = Y
      cuda_safe_call(cudaMemcpyAsync(sX.data_handle(), sY.data_handle(), n * sizeof(int), cudaMemcpyDeviceToDevice, s));
      // Y = TMP
      cuda_safe_call(
        cudaMemcpyAsync(sY.data_handle(), sTMP.data_handle(), n * sizeof(int), cudaMemcpyDeviceToDevice, s));
    };
  }

  ctx.finalize();

  // We have exchanged an odd number of times, so they must be inverted
  for (size_t i = 0; i < n; i++)
  {
    assert(X[i] == 2 * i - 3);
    assert(Y[i] == 3 * i);
  }
}
