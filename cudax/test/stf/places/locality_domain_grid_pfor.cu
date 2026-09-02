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
 * @brief parallel_for over a grid of all locality domains of a device
 *
 * Distributes a blocked parallel_for over every reported domain (the grid
 * adapts to the queried count) and verifies numerical correctness of the
 * result on the host.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  if (ndomains < 2)
  {
    fprintf(stderr, "Device reports a single locality domain: the grid degenerates to one place.\n");
  }

  stream_ctx ctx;

  const int NITER = 8;
  const size_t n  = 16 * 1024 * 1024;

  std::vector<double> X(n);
  std::vector<double> Y(n);

  for (size_t ind = 0; ind < n; ind++)
  {
    X[ind] = sin(1.0 * static_cast<double>(ind));
    Y[ind] = cos(1.0 * static_cast<double>(ind));
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));
  auto handle_Y = ctx.logical_data(make_slice(&Y[0], n));

  auto where = make_locality_domain_grid(dev);
  EXPECT(where.size() == ndomains);

  for (int iter = 0; iter < NITER; iter++)
  {
    ctx.parallel_for(blocked_partition(), where, handle_X.shape(), handle_X.rw(), handle_Y.read())
        ->*[] __device__(size_t i, auto x, auto y) {
              x(i) += y(i);
            };
  }

  ctx.host_launch(handle_X.read(), handle_Y.read())->*[&](auto hX, auto hY) {
    for (size_t ind = 0; ind < n; ind += 1 + ind / 16)
    {
      const double x0       = sin(1.0 * static_cast<double>(ind));
      const double y0       = cos(1.0 * static_cast<double>(ind));
      const double expected = x0 + NITER * y0;
      EXPECT(fabs(hX(ind) - expected) < 0.00001);
      EXPECT(fabs(hY(ind) - y0) < 0.00001);
    }
  };

  ctx.finalize();

  return 0;
}
