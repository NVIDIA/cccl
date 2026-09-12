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
 * @brief Example of reduction implementing using CUB kernels
 */

#include <thrust/device_vector.h>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <int BLOCK_THREADS, typename T>
__global__ void reduce(slice<const T> values, slice<T> partials, size_t nelems)
{
  using namespace cub;
  typedef BlockReduce<T, BLOCK_THREADS> BlockReduceT;

  auto thread_id = BLOCK_THREADS * blockIdx.x + threadIdx.x;

  // Local reduction
  T local_sum = 0;
  for (size_t ind = thread_id; ind < nelems; ind += blockDim.x * gridDim.x)
  {
    local_sum += values(ind);
  }

  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Per-thread tile data
  T result = BlockReduceT(temp_storage).Sum(local_sum);

  if (threadIdx.x == 0)
  {
    partials(blockIdx.x) = result;
  }
}

template <typename Ctx>
void run()
{
  Ctx ctx;

  const size_t N          = 1024 * 16;
  const size_t BLOCK_SIZE = 128;
  const size_t num_blocks = 32;

  int *X, ref_tot;

  X       = new int[N];
  ref_tot = 0;

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = rand() % N;
    ref_tot += X[ind];
  }

  auto values   = ctx.logical_data(X, {N});
  auto partials = ctx.logical_data(shape_of<slice<int>>(num_blocks));
  auto result   = ctx.logical_data(shape_of<slice<int>>(1));

  ctx.task(values.read(), partials.write(), result.write())->*[&](auto stream, auto values, auto partials, auto result) {
    // reduce values into partials
    reduce<BLOCK_SIZE, int><<<num_blocks, BLOCK_SIZE, 0, stream>>>(values, partials, N);

    // reduce partials on a single block into result
    reduce<BLOCK_SIZE, int><<<1, BLOCK_SIZE, 0, stream>>>(partials, result, num_blocks);
  };

  ctx.host_launch(result.read())->*[&](auto p) {
    if (p(0) != ref_tot)
    {
      fprintf(stderr, "INCORRECT RESULT: p sum = %d, ref tot = %d\n", p(0), ref_tot);
      abort();
    }
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
