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
 * @brief An AXPY kernel implemented with a task of the CUDA stream backend
 *
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <mutex>
#include <thread>

using namespace cuda::experimental::stf;

static __global__ void cuda_sleep_kernel(long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }
}

void cuda_sleep(double ms, cudaStream_t stream)
{
  int device;
  cudaGetDevice(&device);

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  long long int clock_cnt = (long long int) (ms * clock_rate);
  cuda_sleep_kernel<<<1, 1, 0, stream>>>(clock_cnt);
}

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < x.size(); i += nthreads)
  {
    y(i) += a * x(i);
  }
}

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

void mytask(stream_ctx ctx, int /*id*/)
{
  // std::cout << "Thread " << id << " is executing.\n";

  const size_t N = 16;

  double alpha = 3.14;

  auto lX = ctx.logical_data<double>(N);
  auto lY = ctx.logical_data<double>(N);

  ctx.parallel_for(lX.shape(), lX.write())->*[] __device__(size_t i, auto x) {
    x(i) = 1.0;
  };

  ctx.task(lY.write())->*[](cudaStream_t, auto) {};

  /* Compute Y = Y + alpha X */
  for (size_t i = 0; i < 10; i++)
  {
    ctx.task(lX.read(), lY.rw())->*[&](cudaStream_t s, auto dX, auto dY) {
      axpy<<<16, 128, 0, s>>>(alpha, dX, dY);
      cuda_sleep(100.0, s);
    };
  }
}

int main()
{
  stream_ctx ctx;

  std::vector<std::thread> threads;
  // Launch 8 threads.
  for (int i = 0; i < 10; ++i)
  {
    threads.emplace_back(mytask, ctx, i);
  }

  // Wait for all threads to complete.
  for (auto& th : threads)
  {
    th.join();
  }

  ctx.finalize();
}
