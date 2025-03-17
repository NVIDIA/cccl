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

/*
 * The goal of this test is to ensure that using read access modes actually
 * results in concurrent tasks
 */

using namespace cuda::experimental::stf;

/**
 * @brief Call `__nanosleep` (potentially repeatedly) to sleep `nanoseconds` nanoseconds. Supports sleep times longer
 * than 4 billion nanoseconds (i.e. 4 seconds).
 *
 * @param nanoseconds how many nanoseconds to sleep
 * @return void
 */
__global__ void nano_sleep(unsigned long long nanoseconds)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  static constexpr auto m = std::numeric_limits<unsigned int>::max();
  for (;;)
  {
    if (nanoseconds > m)
    {
      __nanosleep(m);
      nanoseconds -= m;
    }
    else
    {
      __nanosleep(static_cast<unsigned int>(nanoseconds));
      break;
    }
  }
#else
  const clock_t end = clock() + nanoseconds / (1000000000ULL / CLOCKS_PER_SEC);
  while (clock() < end)
  {
    // busy wait
  }
#endif
}

void run(context& ctx, int NTASKS, int ms)
{
  int dummy[1];
  auto handle = ctx.logical_data(dummy);

  ctx.task().add_deps(handle.rw())->*[](cudaStream_t stream) {
    nano_sleep<<<1, 1, 0, stream>>>(0);
  };

  for (int iter = 0; iter < 10; iter++)
  {
    for (int k = 0; k < NTASKS; k++)
    {
      ctx.task().add_deps(handle.read())->*[&](cudaStream_t stream) {
        nano_sleep<<<1, 1, 0, stream>>>(ms * 1000ULL * 1000ULL);
      };
    }

    ctx.task().add_deps(handle.rw())->*[&](cudaStream_t stream) {
      nano_sleep<<<1, 1, 0, stream>>>(0);
    };
  }

  ctx.finalize();
}

int main(int argc, char** argv)
{
  int NTASKS = 256;
  int ms     = 40;

  if (argc > 1)
  {
    NTASKS = atoi(argv[1]);
  }

  if (argc > 2)
  {
    ms = atoi(argv[2]);
  }

  context ctx;
  run(ctx, NTASKS, ms);
  ctx = graph_ctx();
  run(ctx, NTASKS, ms);
}
