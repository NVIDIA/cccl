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
 * @brief A parallel scan algorithm
 *
 */

#include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__host__ __device__ double X0(int)
{
  //    return sin((double) i);
  return 1.0;
}

int main(int argc, char** argv)
{
  stream_ctx ctx;
  // graph_ctx ctx;

  size_t N = 128 * 1024UL * 1024UL;
  if (argc > 1)
  {
    N = size_t(atoll(argv[1]));
  }

  int check = 0;
  if (argc > 2)
  {
    check = atoi(argv[2]);
  }

  std::vector<double> X(N);

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  auto lX = ctx.logical_data(&X[0], N);

  // No need to move this back to the host if we do not check the result
  if (!check)
  {
    lX.set_write_back(false);
  }

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));
  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  constexpr size_t BLOCK_THREADS = 128;
  constexpr size_t NBLOCKS       = 8;

  auto spec = con<NBLOCKS>(con<BLOCK_THREADS>(), mem(NBLOCKS * sizeof(double)));

  // auto where = exec_place::repeat(exec_place::current_device(), NDEVS);
  auto where = exec_place::current_device();

  ctx.launch(spec, where, lX.rw())->*[=] _CCCL_DEVICE(auto th, auto x) {
    const size_t block_id = th.rank(0);
    const size_t tid      = th.inner().rank();
    // const size_t tid = th.rank(1, 0);

    // Block-wide partials using static allocation
    __shared__ double block_partial_sum[th.static_width(1)];

    // Device-wide partial sums
    slice<double> dev_partial_sum = th.template storage<double>(0);

    /* Thread local prefix-sum */
    const box<1> b = th.apply_partition(shape(x), std::tuple<blocked_partition, blocked_partition>());
    for (size_t i = b.get_begin(0) + 1; i < b.get_end(0); i++)
    {
      x(i) += x(i - 1);
    }
    block_partial_sum[tid] = x(b.get_end(0) - 1);

    th.inner().sync();

    /* Block level : get partials sum accross the different threads */
    if (tid == 0)
    { // rank in scope block is 0
      // Prefix sum on partial sums
      for (size_t i = 1; i < BLOCK_THREADS; i++)
      {
        block_partial_sum[i] += block_partial_sum[i - 1];
      }
      dev_partial_sum[block_id] = block_partial_sum[BLOCK_THREADS - 1];
    }

    /* Reduce partial sums at device level : get sum accross all blocks */
    th.sync();

    if (block_id == 0 && tid == 0)
    { // rank in scope 0
      for (size_t i = 1; i < NBLOCKS; i++)
      {
        dev_partial_sum[i] += dev_partial_sum[i - 1];
        //  printf("SUMMED dev_partial_sum[%ld] = %f\n", i, dev_partial_sum[i]);
      }
    }

    th.sync();

    for (size_t i = b.get_begin(0); i < b.get_end(0); i++)
    {
      if (tid > 0)
      {
        x(i) += block_partial_sum[tid - 1];
      }

      if (block_id > 0)
      {
        x(i) += dev_partial_sum[block_id - 1];
      }
    }
  };

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  float ms = 0;
  cuda_safe_call(cudaEventElapsedTime(&ms, start, stop));

  printf("%s in %f ms (%g GB/s)\n",
         pretty_print_bytes(N * sizeof(double)).c_str(),
         ms,
         double(N * sizeof(double) / 1024 / 1024) / ms);

  if (check)
  {
    fprintf(stderr, "Checking result...\n");
    EXPECT(fabs(X[0] - X0(0)) < 0.00001);
    for (size_t i = 0; i < N; i++)
    {
      if (fabs(X[i] - X[i - 1] - X0(i)) > 0.00001)
      {
        fprintf(stderr, "I %zu X[i] %f (X[i] - X[i-1]) %f expect %f\n", i, X[i], (X[i] - X[i - 1]), X0(i));
      }
      EXPECT(fabs(X[i] - X[i - 1] - X0(i)) < 0.00001);
    }
  }
}
