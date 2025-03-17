//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/__stf/places/cyclic_shape.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

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
  stream_ctx ctx;

  const int N = 128;
  double X[N], Y[N];

  for (int ind = 0; ind < N; ind++)
  {
    X[ind] = X0(ind);
    Y[ind] = Y0(ind);
  }

  const double alpha = 3.14;

  auto handle_X = ctx.logical_data(X, {N});
  auto handle_Y = ctx.logical_data(Y, {N});

  auto number_devices = 4;
  auto all_devs       = exec_place::repeat(exec_place::device(0), number_devices);

  auto spec = par(16 * 4, par(4));
  ctx.launch(spec, all_devs, handle_X.read(), handle_Y.rw())->*[=] _CCCL_DEVICE(auto th, auto x, auto y) {
    // Blocked partition among elements in the outer most level
    auto outer_sh = blocked_partition::apply(shape(x), pos4(th.rank(0)), dim4(th.size(0)));

    // Cyclic partition among elements in the remaining levels
    auto inner_sh = cyclic_partition::apply(outer_sh, pos4(th.inner().rank()), dim4(th.inner().size()));

    for (auto ind : inner_sh)
    {
      y(ind) += alpha * x(ind);
    }
  };

  ctx.host_launch(handle_X.read(), handle_Y.read())->*[=](auto X, auto Y) {
    for (int ind = 0; ind < N; ind++)
    {
      // Y should be Y0 + alpha X0
      // fprintf(stderr, "Y[%ld] = %lf - expect %lf\n", ind, Y(ind), (Y0(ind) + alpha * X0(ind)));
      EXPECT(fabs(Y(ind) - (Y0(ind) + alpha * X0(ind))) < 0.0001);

      // X should be X0
      EXPECT(fabs(X(ind) - X0(ind)) < 0.0001);
    }
  };

  ctx.finalize();
}
