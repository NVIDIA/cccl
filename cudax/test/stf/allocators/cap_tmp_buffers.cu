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

using namespace cuda::experimental::stf;

int main(int, char**)
{
  context ctx;

  const size_t PART_SIZE = 1024;
  const size_t PART_CNT  = 64;

  pooled_allocator_config config;
  config.max_entries_per_place = 8;
  auto fixed_alloc             = block_allocator<pooled_allocator>(ctx, config);

  /* Create a large device buffer which will be used part by part. */
  double* dA;
  cuda_safe_call(cudaMalloc(&dA, PART_SIZE * PART_CNT * sizeof(double)));

  for (size_t p = 0; p < PART_CNT; p++)
  {
    /* Create a logical data from a subset of the existing device buffer */
    auto Ap = ctx.logical_data(make_slice(&dA[p * PART_SIZE], PART_SIZE), data_place::current_device());

    ctx.parallel_for(Ap.shape(), Ap.write()).set_symbol("init_Ap")->*[p, PART_SIZE] __device__(size_t i, auto ap) {
      ap(i) = 1.0 * (i + p * PART_SIZE);
    };

    auto tmp = ctx.logical_data(Ap.shape());
    tmp.set_allocator(fixed_alloc);

    ctx.parallel_for(Ap.shape(), Ap.read(), tmp.write()).set_symbol("set_tmp")->*
      [] __device__(size_t i, auto ap, auto tmp) {
        tmp(i) = 2.0 * ap(i);
      };

    ctx.parallel_for(Ap.shape(), Ap.write(), tmp.read()).set_symbol("update_Ap")
        ->*[] __device__(size_t i, auto ap, auto tmp) {
              ap(i) = tmp(i);
            };
  }

  ctx.finalize();

  cuda_safe_call(cudaFree(dA));
}
