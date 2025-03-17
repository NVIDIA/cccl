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
 * @brief This example illustrates how we can annotate tasks and logical data with debugging symbol
 *
 * CUDASTF_DOT_FILE=axpy.dot build/examples/axpy-annotated
 *
 * # Generate the visualization from this dot file in PDF or PNG format
 * dot -Tpdf axpy.dot -o axpy.pdf
 * dot -Tpng axpy.dot -o axpy.png
 *
 * # Generate visualization with events (for advanced users)
 * CUDASTF_DOT_IGNORE_PREREQS=0 CUDASTF_DOT_FILE=axpy-with-events.dot build/examples/axpy-annotated
 * dot -Tpng axpy-with-events.dot -o axpy-with-events.png
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
  context ctx;
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  double alpha = 3.14;

  auto lX = ctx.logical_data(X).set_symbol("X");
  auto lY = ctx.logical_data(Y).set_symbol("Y");

  /* Compute Y = Y + alpha X */
  ctx.task(lX.read(), lY.rw()).set_symbol("axpy")->*[&](cudaStream_t s, auto dX, auto dY) {
    axpy<<<16, 128, 0, s>>>(alpha, dX, dY);
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }
}
