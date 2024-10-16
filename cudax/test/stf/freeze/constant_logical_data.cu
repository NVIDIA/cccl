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
 * @brief Helper to build constant data based on frozen logical data
 *
 */

#include <cuda/experimental/__stf/utility/constant_logical_data.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  const int N = 16;

  /* Create a constant value */
  auto ld_cst = ctx.logical_data(shape_of<slice<int>>(N));
  ctx.parallel_for(ld_cst.shape(), ld_cst.write())->*[] __device__(size_t i, slice<int> res) {
    res(i) = 18 * i - 9;
  };
  auto cst = constant_logical_data(ctx, mv(ld_cst));

  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = 5 * i - 3;
  }

  auto lX = ctx.logical_data(X).set_symbol("X");

  for (size_t iter = 0; iter < 4; iter++)
  {
    auto cst_slice = cst.get();
    ctx.parallel_for(lX.shape(), lX.rw()).set_symbol("X+=cst")->*[cst_slice] __device__(size_t i, auto x) {
      x(i) += cst_slice(i);
    };

    auto cst2 = run_once()->*[&]() {
      auto ld = ctx.logical_data(shape_of<slice<int>>(N));
      ctx.parallel_for(ld.shape(), ld.write())->*[] __device__(size_t i, slice<int> res) {
        res(i) = 4 * i - 2;
      };
      return constant_logical_data(ctx, mv(ld));
    };

    auto cst2_slice = cst2.get();
    ctx.parallel_for(lX.shape(), lX.rw()).set_symbol("X+=cst2")->*[cst2_slice] __device__(size_t i, auto x) {
      x(i) += cst2_slice(i);
    };
  }

  ctx.finalize();

  for (int i = 0; i < N; i++)
  {
    EXPECT(X[i] == (5 * i - 3) + 4 * (18 * i - 9) + 4 * (4 * i - 2));
  }
}
