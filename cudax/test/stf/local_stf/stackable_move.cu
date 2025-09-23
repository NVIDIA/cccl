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

#include "cuda/experimental/__stf/stackable/stackable_ctx.cuh"

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  auto lA = sctx.logical_data(shape_of<slice<int>>(512));
  lA.set_symbol("A");

  sctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
    a(i) = 42 + 2 * i;
  };

  /* Start to use a graph */
  auto g = sctx.dot_section("foo");
  sctx.push();

  auto lB = sctx.logical_data(lA.shape());
  lB.set_symbol("B");

  auto lA_moved = mv(lA);

  sctx.parallel_for(lB.shape(), lB.write(), lA_moved.read())->*[] __device__(size_t i, auto b, auto a) {
    b(i) = 17 - 3 * a(i);
  };

  auto lC = mv(lB);

  sctx.parallel_for(lC.shape(), lC.rw())->*[] __device__(size_t i, auto b) {
    b(i) *= 2;
  };

  sctx.parallel_for(lA_moved.shape(), lA_moved.rw())->*[] __device__(size_t i, auto a) {
    a(i) *= 3;
  };

  sctx.pop();
  g.end();

  /* Access C in a context below the context where it was created */
  sctx.host_launch(lC.read())->*[](auto b) {
    for (size_t i = 0; i < b.size(); i++)
    {
      EXPECT(b(i) == 2 * (17 - 3 * (42 + 2 * i)));
    }
  };

  sctx.host_launch(lA_moved.read())->*[](auto a) {
    for (size_t i = 0; i < a.size(); i++)
    {
      EXPECT(a(i) == 3 * (42 + 2 * i));
    }
  };

  sctx.finalize();
}
