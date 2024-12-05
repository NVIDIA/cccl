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

__global__ void add(slice<int, 2> s, int val)
{
  size_t tid      = threadIdx.x + blockIdx.x * blockDim.x;
  size_t nthreads = blockDim.x * gridDim.x;

  for (size_t j = 0; j < s.extent(1); j++)
  {
    for (size_t i = tid; i < s.extent(0); i += nthreads)
    {
      s(i, j) += val;
    }
  }
}

int main()
{
  stream_ctx ctx;

  int array[6] = {0, 1, 2, 3, 4, 5};

  // auto handle = ctx.logical_data(slice<int, 2>(&array[0], std::tuple{ 2, 3 }, 2));
  auto handle = ctx.logical_data(make_slice(&array[0], std::tuple{2, 3}, 2));

  auto redux_op = std::make_shared<slice_reduction_op_sum<int, 2>>();

  ctx.task(handle.relaxed(redux_op))->*[](auto stream, auto s) {
    add<<<32, 32, 0, stream>>>(s, 42);
  };

  ctx.task(exec_place::host, handle.read())->*[](auto stream, auto s) {
    cuda_safe_call(cudaStreamSynchronize(stream));

    for (size_t j = 0; j < s.extent(1); j++)
    {
      for (size_t i = 0; i < s.extent(0); i++)
      {
        // fprintf(stderr, "%d\t", s(i, j));
      }
      // fprintf(stderr, "\n");
    }
  };

  ctx.finalize();
}
