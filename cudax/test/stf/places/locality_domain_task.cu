//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief AXPY tasks submitted to every locality domain of a device
 *
 * Round-robins tasks over all reported domains (adapting to the queried
 * count) and checks numerical correctness of the result, exercising both
 * per-domain execution and transfers to/from domain-affine data places.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  size_t n = x.extent(0);
  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    y(ind) += a * x(ind);
  }
}

int main()
{
  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  stream_ctx ctx;
  const double alpha = 2.0;

  const int NITER = 4 * static_cast<int>(ndomains);
  const int n     = 1024;

  double X[n], Y[n];
  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));
  auto handle_Y = ctx.logical_data(make_slice(&Y[0], n));

  // Visit every domain the same number of times
  for (int iter = 0; iter < NITER; iter++)
  {
    const int domain_id = iter % static_cast<int>(ndomains);
    ctx.task(exec_place::locality_domain(dev, domain_id), handle_X.read(), handle_Y.rw())
        ->*[&](cudaStream_t stream, auto dX, auto dY) {
              axpy<<<16, 128, 0, stream>>>(alpha, dX, dY);
            };
  }

  ctx.host_launch(handle_X.read(), handle_Y.read())->*[&](auto hX, auto hY) {
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(hX(ind) - 1.0 * ind) < 0.00001);
      EXPECT(fabs(hY(ind) - (2.0 * ind - 3.0) - NITER * alpha * hX(ind)) < 0.00001);
    }
  };

  ctx.finalize();

  return 0;
}
