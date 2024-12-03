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

__global__ void add(int* ptr, int value)
{
  *ptr = *ptr + value;
}

int main()
{
  stream_ctx ctx;

  int a         = 17;
  auto handle   = ctx.logical_data(make_slice(&a, 1));
  auto redux_op = std::make_shared<slice_reduction_op_sum<int>>();
  ctx.task(handle.relaxed(redux_op))->*[](auto stream, auto s) {
    add<<<1, 1, 0, stream>>>(s.data_handle(), 42);
  };

  ctx.task(handle.rw())->*[](auto stream, auto s) {
    add<<<1, 1, 0, stream>>>(s.data_handle(), 1);
  };

  ctx.task(exec_place::host, handle.read())->*[](auto stream, auto s) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    EXPECT(s(0) == 17 + 42 + 1);
  };

  ctx.finalize();
}
