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
 * @brief Experiment to see if we can read data generated in a nested context
 *
 */

#include <cuda/experimental/stf.cuh>

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  auto lA = sctx.logical_data(shape_of<slice<int>>(512));
  lA.set_symbol("A");

  sctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
    a(i) = 42 + 2 * i;
  };

  auto ltoken = sctx.token();

  /* Start to use a graph */
  sctx.push();

  lA.push(access_mode::rw);
  ltoken.push(access_mode::rw);

  auto ltoken2 = sctx.token();

  auto lB = sctx.logical_data(lA.shape());
  lB.set_symbol("B");

  sctx.parallel_for(lB.shape(), lB.write(), lA.read(), ltoken.rw())->*[] __device__(size_t i, auto b, auto a) {
    b(i) = 17 - 3 * a(i);
  };

  auto lC = mv(lB);

  sctx.parallel_for(lC.shape(), lC.rw())->*[] __device__(size_t i, auto b) {
    b(i) *= 2;
  };

  sctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) *= 3;
  };

  sctx.pop();

  /* Access C in a context below the context where it was created */
  sctx.host_launch(lC.read(), ltoken2.read())->*[](auto b, auto) {
    for (size_t i = 0; i < b.size(); i++)
    {
      EXPECT(b(i) == 2 * (17 - 3 * (42 + 2 * i)));
    }
  };

  sctx.host_launch(lA.read(), ltoken.read())->*[](auto a, auto) {
    for (size_t i = 0; i < a.size(); i++)
    {
      EXPECT(a(i) == 3 * (42 + 2 * i));
    }
  };

  sctx.finalize();
}
