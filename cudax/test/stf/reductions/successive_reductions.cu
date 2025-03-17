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

using scalar_t = slice_stream_interface<int, 1>;

template <typename T>
__global__ void set_value(T* addr, T val)
{
  *addr = val;
}

template <typename T>
__global__ void check_value_and_reset(T* addr, T val, T reset_val)
{
  assert(*addr == val);
  *addr = reset_val;
}

template <typename T>
__global__ void add_val(T* inout_addr, T val)
{
  *inout_addr += val;
}

int main()
{
  stream_ctx ctx;
  const int N = 4;

  auto var_handle = ctx.logical_data(shape_of<slice<int>>(1));
  var_handle.set_symbol("var");

  auto redux_op = std::make_shared<slice_reduction_op_sum<int>>();

  const int niters = 4;
  for (int iter = 0; iter < niters; iter++)
  {
    // We add i (total = N(N-1)/2 + initial_value)
    for (int i = 0; i < N; i++)
    {
      ctx.task(var_handle.relaxed(redux_op))->*[=](cudaStream_t stream, auto d_var) {
        add_val<<<1, 1, 0, stream>>>(d_var.data_handle(), i);
        cuda_safe_call(cudaGetLastError());
      };
    }

    // Check that we have the expected value, and reset it so that we can perform another reduction
    ctx.task(var_handle.rw())->*[=](cudaStream_t stream, auto d_var) {
      int expected = (N * (N - 1)) / 2;
      check_value_and_reset<<<1, 1, 0, stream>>>(d_var.data_handle(), expected, 0);
    };
  }

  ctx.finalize();
}
