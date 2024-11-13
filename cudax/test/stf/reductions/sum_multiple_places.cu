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

template <typename T>
__global__ void add_val(slice<T> inout, T val)
{
  inout(0) += val;
}

int main()
{
  stream_ctx ctx;

  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  const int N = 4;

  int var = 42;

  auto var_handle = ctx.logical_data(make_slice(&var, 1));
  var_handle.set_symbol("var");

  auto redux_op = std::make_shared<slice_reduction_op_sum<int>>();

  // We add i twice (total = N(N-1) + initial_value)
  for (int i = 0; i < N; i++)
  {
    // device
    for (int d = 0; d < ndevs; d++)
    {
      ctx.task(exec_place::device(d), var_handle.relaxed(redux_op))->*[=](cudaStream_t s, auto var) {
        add_val<int><<<1, 1, 0, s>>>(var, i);
      };
    }

    // host
    ctx.host_launch(var_handle.relaxed(redux_op))->*[=](auto var) {
      var(0) += i;
    };
  }

  // Check result
  ctx.host_launch(var_handle.read())->*[&](auto var) {
    int expected = 42 + (N * (N - 1)) / 2 * (ndevs + 1);
    EXPECT(var(0) == expected);
  };

  ctx.finalize();
}
