//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/allocators/adapters.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  double* d_ptrA;
  const size_t N     = 128 * 1024;
  const size_t NITER = 10;

  // User allocated memory
  cuda_safe_call(cudaMalloc(&d_ptrA, N * sizeof(double)));

  async_resources_handle handle;

  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  for (size_t i = 0; i < NITER; i++)
  {
    graph_ctx ctx(stream, handle);

    auto wrapper = stream_adapter(ctx, stream);
    ctx.set_allocator(block_allocator<buddy_allocator>(ctx, wrapper.allocator()));

    auto A = ctx.logical_data(make_slice(d_ptrA, N), data_place::current_device());

    for (size_t k = 0; k < 4; k++)
    {
      auto tmp  = ctx.logical_data(A.shape());
      auto tmp2 = ctx.logical_data(A.shape());
      // Test device and managed memory
      ctx.parallel_for(A.shape(), A.read(), tmp.write(), tmp2.write(data_place::managed))
          ->*[] __device__(size_t i, auto a, auto tmp, auto tmp2) {
                tmp(i)  = a(i);
                tmp2(i) = a(i);
              };
    }

    ctx.finalize();

    wrapper.clear();
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
}
