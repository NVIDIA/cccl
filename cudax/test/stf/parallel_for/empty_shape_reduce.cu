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
 * @brief Test reduce access mode when we encounter an empty shape to ensure we do initialize values
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename context_t>
void run()
{
  context_t ctx;
  auto lsum = ctx.logical_data(shape_of<scalar_view<int>>());
  auto lmax = ctx.logical_data(shape_of<scalar_view<int>>());

  ctx.parallel_for(box(0), lsum.reduce(reducer::sum<int>{}), lmax.reduce(reducer::maxval<int>{}))
      ->*[] __device__(size_t, auto&, auto&) {
            // This is never going to be called because this is an empty shape
          };

  auto res_sum = ctx.wait(lsum);
  auto res_max = ctx.wait(lmax);

  ctx.finalize();

  _CCCL_ASSERT(res_sum == 0, "Invalid result");
  _CCCL_ASSERT(res_max == ::std::numeric_limits<int>::lowest(), "Invalid result");
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
