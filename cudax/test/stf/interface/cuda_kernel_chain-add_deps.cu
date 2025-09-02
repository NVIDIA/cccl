//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Example of task implementing a chain of CUDA kernels with dynamic dependencies (add_deps)
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < x.size(); i += nthreads)
  {
    y(i) += a * x(i);
  }
}

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

  double alpha = 3.14;
  double beta  = 4.5;
  double gamma = -4.1;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  /* Compute Y = Y + alpha X, Y = Y + beta X and then  Y = Y + gamma X */
  auto t = ctx.cuda_kernel_chain();
  t.add_deps(lX.read());
  t.add_deps(lY.rw());
  t->*[&]() {
    auto dX = t.template get<slice<double>>(0);
    auto dY = t.template get<slice<double>>(1);
    // clang-format off
    return std::vector<cuda_kernel_desc> {
        { axpy, 16, 128, 0, alpha, dX, dY },
        { axpy, 16, 128, 0, beta, dX, dY },
        { axpy, 16, 128, 0, gamma, dX, dY }
    };
    // clang-format on
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + (alpha + beta + gamma) * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }
}
