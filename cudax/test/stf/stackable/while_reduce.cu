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
 * @brief Check that reduction access mode in parallel_for behaves as expected
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;

  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));
  ctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto dA) {
    dA(i) = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    auto lsum = ctx.logical_data(shape_of<scalar_view<int>>());

    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto dA) {
      dA(i)++;
    };

    ctx.parallel_for(lA.shape(), lA.read(), lsum.reduce(reducer::sum<int>()))
        ->*[] __device__(size_t i, auto dA, int& sum) {
              sum += dA(i);
            };

    while_guard.update_cond(lsum.read())->*[] __device__(auto sum) {
      bool converged = (*sum > 4096);
      return !converged;
    };
  }

  ctx.finalize();
}
