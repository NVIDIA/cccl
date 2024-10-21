//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

#include <cstdio>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  int array[128];
  for (size_t i = 0; i < 128; i++)
  {
    array[i] = i;
  }

  auto A = ctx.logical_data(array);

  ctx.parallel_for(A.shape(), A.rw())->*[] __device__(size_t i, auto a) {
    a(i) += 4;
  };

  ctx.finalize();

  for (size_t i = 0; i < 128; i++)
  {
    printf("array[%ld] = %d\n", i, array[i]);
  }
  return 0;
}
