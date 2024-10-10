//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/place_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
#if CUDA_VERSION < 12040
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else

  context ctx;

  auto lX = ctx.logical_data<int>(size_t(32 * 1024 * 1024));

  ctx.parallel_for(lX.shape(), lX.write())->*[] __device__(size_t i, auto x) {
    x(i) = 3 * i - 7;
  };

  for (auto& sub_place :
       place_partition(ctx.async_resources(), exec_place::current_device(), place_partition_scope::green_context))
  {
    for (size_t i = 0; i < 4; i++)
    {
      ctx.parallel_for(sub_place, lX.shape(), lX.rw())->*[] __device__(size_t i, auto x) {
        x(i) += 1;
      };
    }
  }

  ctx.finalize();
#endif
}
