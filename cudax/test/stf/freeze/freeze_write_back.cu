//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//!
//! \brief Ensure write-back is working on logical data alias made by freezing another one

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    array[i] = 2 - i * i;
  }

  auto lA = ctx.logical_data(array).set_symbol("A");

  auto stream = ctx.pick_stream();

  graph_ctx gctx(stream);

  // Create an alias for lA in the graph by freezing it and creating a new
  // local logical data in the graph.
  auto fa   = ctx.freeze(lA, access_mode::rw, data_place::current_device());
  auto inst = fa.get(data_place::current_device(), stream);
  auto glA  = gctx.logical_data(inst, data_place::current_device());

  gctx.parallel_for(glA.shape(), glA.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 4 * i;
  };

  // force to move to a different place, and probably to allocate another copy
  // on the host. This tests if the write-back mechanism works from the host to
  // the device when destroying the alias logical data glA.
  gctx.host_launch(glA.rw())->*[](auto a) {
    for (size_t i = 0; i < 1024; i++)
    {
      a(i) *= 2;
    }
  };

  gctx.finalize();

  fa.unfreeze(stream);

  ctx.finalize();

  for (size_t i = 0; i < 1024; i++)
  {
    EXPECT(array[i] == 2 * (2 - i * i + 4 * i));
  }
}
