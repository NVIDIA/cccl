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
 * @brief Illustrate how we can use frozen data to initialize constant data
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  /* Create a piece of data that can be use many times without further synchronizations */
  auto buffer = ctx.logical_data(shape_of<slice<double, 2>>(128, 64)).set_symbol("buffer");
  ctx.parallel_for(buffer.shape(), buffer.write())->*[] __device__(size_t i, size_t j, auto b) {
    b(i, j) = sin(-1.0 * i) + cos(2.0 * j);
  };

  auto frozen_buffer = ctx.freeze(buffer);

  auto h_buf = frozen_buffer.get(data_place::host()).first;
  auto d_buf = frozen_buffer.get(data_place::current_device()).first;

  cuda_safe_call(cudaStreamSynchronize(ctx.fence()));

  auto lX = ctx.logical_data(buffer.shape()).set_symbol("X");
  ctx.parallel_for(lX.shape(), lX.write()).set_symbol("X=buf")->*[d_buf] __device__(size_t i, size_t j, auto x) {
    x(i, j) = d_buf(i, j);
  };

  // Host-side check: host work goes through host_launch with an explicit loop, parallel_for
  // being a device-only construct.
  ctx.host_launch(lX.read()).set_symbol("check buf")->*[h_buf](auto x) {
    for (size_t i = 0; i < x.extent(0); i++)
    {
      for (size_t j = 0; j < x.extent(1); j++)
      {
        EXPECT(fabs(x(i, j) - h_buf(i, j)) < 0.0001);
      }
    }
  };

  // Make sure all tasks are done before unfreezing
  frozen_buffer.unfreeze(ctx.fence());

  ctx.finalize();
}
