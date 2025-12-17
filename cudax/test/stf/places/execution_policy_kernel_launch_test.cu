//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! file
//! !brief Check that multi-level launch specification are fulfilled

#include <cuda/experimental/stf.cuh>

#include <cassert>
#include <iostream>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  // Create a 3-level thread hierarchy specification that would expose the bug:
  // Level 0: only 1 device to run on CI
  // Level 1: 4 blocks per device (width 4)
  // Level 2: 64 threads per block (width 64)
  //
  auto spec = par(hw_scope::device, 1, con<4>(hw_scope::block, con<64>(hw_scope::thread)));

  int test_result    = 0;
  auto l_test_result = ctx.logical_data(make_slice(&test_result, 1));

  ctx.launch(spec, exec_place::current_device(), l_test_result.rw())->*[] __device__(auto th, auto result) {
    if (th.rank() == 0)
    {
      bool level0_correct = (th.size(0) == 1); // device level
      bool level1_correct = (th.size(1) == 1 * 4) && (gridDim.x == 4); // blocks per device
      bool level2_correct = (th.size(2) == 1 * 4 * 64) && (blockDim.x == 64); // threads per block

      // Set test result based on whether all levels are correct
      result[0] = level0_correct && level1_correct && level2_correct ? 1 : 0;
    }
  };

  ctx.finalize();

  if (test_result != 1)
  {
    fprintf(stderr, "FAIL: Hierarchy dimensions are incorrect!\n");
    return 1;
  }

  return 0;
}
