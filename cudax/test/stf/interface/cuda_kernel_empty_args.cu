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
 * @brief Make sure we can compile with no arguments in CUDA kernel
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

int main()
{
  context ctx    = graph_ctx();
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  // Ensure this works without arguments in the kernel
  ctx.cuda_kernel(lX.read(), lY.rw())->*[&](auto, auto) {
    return cuda_kernel_desc{dummy, 16, 128, 0};
  };

  ctx.finalize();
}
