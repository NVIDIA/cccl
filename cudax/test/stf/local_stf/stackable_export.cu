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

  lA.push(access_mode::read);

  auto lB = sctx.logical_data(lA.shape());
  lB.set_symbol("B");

  auto lC = sctx.logical_data_no_export(lA.shape());
  // auto lC = sctx.logical_data(lA.shape());
  lC.set_symbol("C");

  sctx.parallel_for(lB.shape(), lB.write(), lA.read(), lC.write())->*[] __device__(size_t i, auto b, auto a, auto c) {
    b(i) = 17 - 3 * a(i);
    c(i) = b(i);
  };

  sctx.parallel_for(lB.shape(), lB.rw())->*[] __device__(size_t i, auto b) {
    b(i) *= 2;
  };

  sctx.pop();
  g.end();

  /* Access B in a context below the context where it was created */
  sctx.host_launch(lB.read())->*[](auto b) {
    for (size_t i = 0; i < b.size(); i++)
    {
      EXPECT(b(i) == 2 * (17 - 3 * (42 + 2 * i)));
    }
  };

  sctx.finalize();
}
