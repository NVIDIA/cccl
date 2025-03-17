//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

/*
 * The goal of this test is to ensure that using read access modes actually
 * results in concurrent tasks
 */

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

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cuda_safe_call(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0));
  long long int clock_cnt = (long long int) (ms * clock_rate);

  graph_ctx ctx;

  int dummy[1];
  auto handle = ctx.logical_data(dummy);

  ctx.task(handle.rw())->*[](cudaGraph_t graph, auto /*unused*/) {
    cudaGraphNode_t n;
    cuda_safe_call(cudaGraphAddEmptyNode(&n, graph, nullptr, 0));
  };

  for (int iter = 0; iter < 10; iter++)
  {
    for (int k = 0; k < NTASKS; k++)
    {
      ctx.task(handle.read())->*[&](cudaStream_t stream, auto /*unused*/) {
        cuda_sleep_kernel<<<1, 1, 0, stream>>>(clock_cnt);
      };
    }

    ctx.task(handle.rw())->*[&](cudaGraph_t graph, auto /*unused*/) {
      cudaGraphNode_t n;
      cuda_safe_call(cudaGraphAddEmptyNode(&n, graph, nullptr, 0));
    };
  }

  ctx.submit();

  if (argc > 3)
  {
    std::cout << "Generating DOT output in " << argv[3] << std::endl;
    ctx.print_to_dot(argv[3]);
  }

  ctx.finalize();
}
