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
 * @brief Experiment to see if we can read data generated in a nested context
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void kernel(slice<int> b, long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }

  size_t n   = b.size();
  int i      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  while (i < n)
  {
    b(i) = 17 - 3 * i;
    i += stride;
  }
}

int main()
{
  int device;
  cudaGetDevice(&device);

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  double ms               = 500;
  long long int clock_cnt = (long long int) (ms * clock_rate);

  stackable_ctx sctx;

  auto lB = sctx.logical_data(shape_of<slice<int>>(1024));
  sctx.push();
  lB.push(access_mode::write);

  lB.set_symbol("B");

  sctx.task(lB.write())->*[clock_cnt](cudaStream_t stream, auto b) {
    kernel<<<32, 4, 0, stream>>>(b, clock_cnt);
  };

  sctx.pop();

  /* Access B in a context below the context where it was created */
  sctx.host_launch(lB.read())->*[](auto b) {
    for (size_t i = 0; i < b.size(); i++)
    {
      EXPECT(b(i) == (17 - 3 * i));
    }
  };

  sctx.finalize();
}
