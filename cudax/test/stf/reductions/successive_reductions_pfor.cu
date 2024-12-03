//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/interfaces/slice_reduction_ops.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;
  const int N = 4;

  auto var_handle = ctx.logical_data(shape_of<slice<int>>(1));
  var_handle.set_symbol("var");

  auto op = std::make_shared<slice_reduction_op_sum<int>>();

  const int niters = 4;
  for (int iter = 0; iter < niters; iter++)
  {
    // We add i (total = N(N-1)/2 + initial_value)
    for (int i = 0; i < N; i++)
    {
      ctx.parallel_for(var_handle.shape(), var_handle.relaxed(op))->*[=] _CCCL_DEVICE(size_t ind, auto d_var) {
        atomicAdd(d_var.data_handle(), i);
      };
    }

    // Check that we have the expected value, and reset it so that we can perform another reduction
    ctx.parallel_for(var_handle.shape(), var_handle.rw())->*[=] _CCCL_DEVICE(size_t ind, auto d_var) {
      assert(d_var(0) == (N * (N - 1)) / 2);
      d_var(0) = 0;
    };
  }

  ctx.finalize();
}
