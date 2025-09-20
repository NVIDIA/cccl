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
 * @brief Experiment with local context and temporary logical data
 *
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename T>
static __global__ void kernel_set(T* addr, T val)
{
  printf("SETTING ADDR %p at %d\n", addr, val);
  *addr = val;
}

int main()
{
  stackable_ctx ctx;
  auto lA = ctx.logical_data(shape_of<slice<int>>(1024));

  ctx.task(lA.write())->*[](cudaStream_t stream, auto a) {
    kernel_set<<<1, 1, 0, stream>>>(a.data_handle(), 42);
  };

  {
    auto scope = ctx.graph_scope();

    // Create temporary data in nested context
    auto temp = ctx.logical_data(shape_of<slice<int>>(1024));

    ctx.parallel_for(lA.shape(), temp.write(), lA.read())->*[] __device__(size_t i, auto temp, auto a) {
      // Copy data and modify
      temp(i) = a(i) * 2;
    };

    ctx.parallel_for(lA.shape(), lA.write(), temp.read())->*[] __device__(size_t i, auto a, auto temp) {
      // Copy back
      a(i) = temp(i) + 1;
    };

    // temp automatically cleaned up when scope ends
  }

  ctx.finalize();
}
