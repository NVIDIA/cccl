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

__global__ void check_val(const int* ptr, int expected)
{
  assert(*ptr == expected);
}

/*
 * This test ensures that we can reconstruct a piece of data where there are
 * multiple "shared" instances, and "redux" instances
 */
int main()
{
  stream_ctx ctx;

  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  if (ndevs < 2)
  {
    fprintf(stderr, "Skipping test: need at least 2 devices.\n");
    return 0;
  }

  auto redux_op = std::make_shared<slice_reduction_op_sum<int>>();

  int a = 17;

  // init op (17)
  auto handle = ctx.logical_data(make_slice(&a, 1));

  // RW dev0 (18)
  ctx.task(exec_place::device(0), handle.rw())->*[](auto stream, auto s) {
    add<<<1, 1, 0, stream>>>(s.data_handle(), 1);
  };

  // READ dev1 (18)
  ctx.task(exec_place::device(1), handle.read())->*[](auto stream, auto s) {
    check_val<<<1, 1, 0, stream>>>(s.data_handle(), 18);
  };

  // REDUX dev1 (18 + 42)
  ctx.task(exec_place::device(1), handle.relaxed(redux_op))->*[](auto stream, auto s) {
    add<<<1, 1, 0, stream>>>(s.data_handle(), 42);
  };

  // READ dev0 (18 + 42)
  ctx.task(exec_place::device(0), handle.read())->*[](auto stream, auto s) {
    check_val<<<1, 1, 0, stream>>>(s.data_handle(), 18 + 42);
  };

  // READ
  ctx.task(exec_place::host, handle.read())->*[](auto stream, auto s) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    EXPECT(s(0) == 18 + 42);
    // printf("VALUE %d expected %d\n", s(0), 18 + 42);
  };

  ctx.finalize();
}
