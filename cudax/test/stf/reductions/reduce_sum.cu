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
 * @brief Test reduce access mode
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;
  auto lsum = ctx.logical_data(shape_of<scalar_view<size_t>>());

  size_t N = 100000;

  ctx.parallel_for(box(N), lsum.reduce(reducer::sum<size_t>{}))->*[] __device__(size_t i, auto& sum) {
    sum++;
  };

  size_t res_sum = ctx.wait(lsum);

  ctx.finalize();

  _CCCL_ASSERT(res_sum == N, "Invalid result");
}
