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
 * @brief Experiment with local context nesting
 *
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx sctx;

  int array[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    array[i] = 1 + i * i;
  }

  auto lC = sctx.logical_data(array);

  auto lA = sctx.logical_data(lC.shape());
  lA.set_symbol("A");

  auto lA2 = sctx.logical_data(shape_of<slice<int>>(1024));
  lA2.set_symbol("A2");

  sctx.parallel_for(lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
    a(i) = 42 + 2 * i;
  };

  /* Create nested graph */
  {
    stackable_ctx::graph_scope scope{sctx};

    auto lB = sctx.logical_data(shape_of<slice<int>>(512));
    lB.set_symbol("B");

    sctx.parallel_for(lB.shape(), lB.write())->*[] __device__(size_t i, auto b) {
      b(i) = 17 - 3 * i;
    };

    sctx.parallel_for(lA2.shape(), lA2.write())->*[] __device__(size_t i, auto a2) {
      a2(i) = 5 * i + 4;
    };

    sctx.parallel_for(lB.shape(), lA.read(), lB.rw())->*[] __device__(size_t i, auto a, auto b) {
      b(i) += a(i);
    };

    sctx.parallel_for(lB.shape(), lB.read(), lC.rw())->*[] __device__(size_t i, auto b, auto c) {
      c(i) += b(i);
    };

    sctx.print_logical_data_summary();
  }

  sctx.host_launch(lA2.read())->*[](auto a2) {
    for (size_t i = 0; i < a2.size(); i++)
    {
      EXPECT(a2(i) == 5 * i + 4);
    }
  };

  // Do the same check in another graph
  {
    stackable_ctx::graph_scope scope{sctx};
    lA2.push(access_mode::read);
    sctx.host_launch(lA2.read())->*[](auto a2) {
      for (size_t i = 0; i < a2.size(); i++)
      {
        EXPECT(a2(i) == 5 * i + 4);
      }
    };
  }

  sctx.finalize();
}
