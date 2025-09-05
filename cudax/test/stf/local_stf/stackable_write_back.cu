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

#include <cuda/experimental/stf.cuh>

#include "cuda/experimental/__stf/utility/stackable_ctx.cuh"

using namespace cuda::experimental::stf;

int X0(int i)
{
  return 17 * i + 45;
}

int main()
{
  stackable_ctx sctx;

  int array[1024];
  int array2[1024];
  int array3[1024];
  int array4[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    array[i]  = 1 + i * i;
    array2[i] = 4 - i;
    array3[i] = 19 + 5 * i;
    array4[i] = 2 - i * i;
  }

  auto lA  = sctx.logical_data(array).set_symbol("A");
  auto lA2 = sctx.logical_data(array2).set_symbol("A2");
  auto lA3 = sctx.logical_data(array3).set_symbol("A3");
  auto lA4 = sctx.logical_data(array4).set_symbol("A4");

  /* Start to use a graph */
  sctx.push();

  sctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 2 * i;
  };

  // Copy the (stackable) logical data, which is not a deep copy so modifying
  // lA2_cpy should also modify lA2.
  auto lA2_cpy = lA2;
  sctx.parallel_for(lA2_cpy.shape(), lA2_cpy.rw())->*[] __device__(size_t i, auto a2) {
    a2(i) += 4 * i;
  };

  auto lA3_mv = mv(lA3);
  sctx.parallel_for(lA3_mv.shape(), lA3_mv.rw())->*[] __device__(size_t i, auto a3) {
    a3(i) += i;
  };

  sctx.parallel_for(lA4.shape(), lA4.rw())->*[] __device__(size_t i, auto a4) {
    a4(i) += 4 * i;
  };

  // force to move to a different place, and probably to allocate another copy on the host
  // XXX (it seems that this write is not written back to the copy of array4 on the device when popping before we write
  // back ?)
  sctx.host_launch(lA4.rw())->*[](auto a4) {
    for (size_t i = 0; i < 1024; i++)
    {
      a4(i) *= 2;
    }
  };

  sctx.pop();

  sctx.finalize();

  // Ensure the write-back mechanism was effective
  for (size_t i = 0; i < 1024; i++)
  {
    EXPECT(array[i] == (1 + i * i) + 2 * i);
  }

  for (size_t i = 0; i < 1024; i++)
  {
    EXPECT(array2[i] == (4 - i) + 4 * i);
  }

  for (size_t i = 0; i < 1024; i++)
  {
    EXPECT(array3[i] == (19 + 5 * i) + i);
  }

  for (size_t i = 0; i < 1024; i++)
  {
    EXPECT(array4[i] == 2 * (2 - i * i + 4 * i));
  }
}
