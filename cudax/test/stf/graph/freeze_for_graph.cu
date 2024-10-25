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
 * @brief Ensure temporary data are destroyed
 *
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

int X0(int i)
{
  return 17 * i + 45;
}

__global__ void dummy() {}

int main()
{
  stream_ctx ctx;
  const int N = 16;
  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  auto lX = ctx.logical_data(X);

  auto fX = ctx.freeze(lX, access_mode::rw, data_place::current_device());

  auto stream = ctx.pick_stream();

  graph_ctx gctx(stream);

  auto frozen_X = fX.get(data_place::current_device(), stream);
  auto lX_alias = gctx.logical_data(frozen_X, data_place::current_device());

  auto lY = gctx.logical_data(lX.shape());

  gctx.parallel_for(lX.shape(), lX_alias.read(), lY.write())->*[] __device__(size_t i, auto x, auto y) {
    y(i) = x(i);
  };

  gctx.parallel_for(lX.shape(), lX_alias.write(), lY.read())->*[] __device__(size_t i, auto x, auto y) {
    x(i) = y(i) + 2;
  };

  gctx.finalize();

  fX.unfreeze(stream);

  ctx.host_launch(lX.read())->*[](auto x) {
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
      EXPECT(x(i) == X0(i) + 2);
    }
  };

  ctx.finalize();
}
