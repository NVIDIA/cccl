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
 * @brief Toy example to reproduce the asynchrony of a 1F1B pipeline
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void forward(slice<int>, long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }
}

__global__ void backward(slice<int>, long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }
}

int main(int argc, char** argv)
{
  context ctx;
  // Use a graph context if the second argument is set and not null
  if (argc > 2 && atoi(argv[2]))
  {
    ctx = graph_ctx();
  }

  int device;
  cudaGetDevice(&device);

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  int f_grid_size, f_block_size;
  std::tie(f_grid_size, f_block_size) = reserved::compute_occupancy(forward);

  int b_grid_size, b_block_size;
  std::tie(b_grid_size, b_block_size) = reserved::compute_occupancy(backward);

  int factor = 1;
  if (argc > 1)
  {
    factor = atoi(argv[1]);
  }

  size_t num_batches = 8 * factor;
  int num_devs       = 8;
  int real_devs;
  cuda_safe_call(cudaGetDeviceCount(&real_devs));

  std::vector<logical_data<slice<int>>> data;

  for (size_t b = 0; b < num_batches; b++)
  {
    auto batch_data = ctx.logical_data(shape_of<slice<int>>(1024));
    data.push_back(batch_data);

    ctx.task(exec_place::device(0), data[b].write())->*[](cudaStream_t, auto) {
      // Init ...
    };
  }

  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));

  size_t niter = 10;

  for (size_t iter = 0; iter < niter; iter++)
  {
    for (size_t b = 0; b < num_batches; b++)
    {
      for (int d = 0; d < num_devs; d++)
      {
        ctx.task(exec_place::device(d % real_devs), data[b].rw())->*[=](cudaStream_t s, auto bd) {
          int ms                  = 10;
          long long int clock_cnt = (long long int) (ms * clock_rate / factor);
          forward<<<f_grid_size, f_block_size, 0, s>>>(bd, clock_cnt);
        };
      }
      //        }
      //
      //        for (size_t b = 0; b < num_batches; b++) {
      for (int d = num_devs; d-- > 0;)
      {
        ctx.task(exec_place::device(d % real_devs), data[b].rw())->*[=](cudaStream_t s, auto bd) {
          int ms                  = 20;
          long long int clock_cnt = (long long int) (ms * clock_rate / factor);
          backward<<<b_grid_size, b_block_size, 0, s>>>(bd, clock_cnt);
        };
      }
    }

    /* We introduce a fence because the actual pipeline would introduce
     * some all to all communication to update coefficients */
    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
  }

  ctx.finalize();
  return 0;
}
