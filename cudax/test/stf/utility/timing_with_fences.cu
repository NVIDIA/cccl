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

template <typename Ctx_t>
void run(int NTASKS, int ms)
{
  Ctx_t ctx;

  int dummy[1];
  auto handle = ctx.logical_data(dummy);

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));

  // warm-up
  ctx.task(handle.rw())->*[ms](cudaStream_t stream, auto) {
    cuda_sleep(ms, stream);
  };

  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  for (int iter = 0; iter < NTASKS; iter++)
  {
    ctx.task(handle.rw())->*[ms](cudaStream_t stream, auto) {
      cuda_sleep(ms, stream);
    };
  }

  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  [[maybe_unused]] float elapsed;
  cuda_safe_call(cudaEventElapsedTime(&elapsed, start, stop));

  [[maybe_unused]] float expected = 1.0f * NTASKS * ms;

  /* We cannot really expect this measurement to be accurate because the
   * thread(s) executing the code might be preempted on a system with a high load
   * (as during unit tests). So the best we can expect is that the elapsed time
   * is larger than the sleep time, but event the timer on the GPU is not
   * perfectly accurate so we do not make any strict assumptions about the
   * test, and just keep this test to demonstrate how to use the mechanisms,
   * and ensure they are functional . */
  // EXPECT(elapsed >= expected);
}

int main(int argc, char** argv)
{
  int NTASKS = 25;
  int ms     = 200;

  if (argc > 1)
  {
    NTASKS = atoi(argv[1]);
  }

  if (argc > 2)
  {
    ms = atoi(argv[2]);
  }

  run<context>(NTASKS, ms);
  run<stream_ctx>(NTASKS, ms);
  run<graph_ctx>(NTASKS, ms);
}
