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
 * @brief Ensure we can create graph contexts many times
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  for (size_t i = 0; i < 10240; i++)
  {
    graph_ctx ctx;
    auto lA = ctx.logical_data(shape_of<slice<size_t>>(64));
    ctx.launch(lA.write())->*[] _CCCL_DEVICE(auto t, slice<size_t> A) {
      for (auto i : t.apply_partition(shape(A)))
      {
        A(i) = 2 * i;
      }
    };
    ctx.finalize();
  }
}
