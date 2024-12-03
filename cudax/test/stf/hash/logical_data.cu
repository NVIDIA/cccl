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
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename S>
__global__ void inc_kernel(S sA)
{
  sA(threadIdx.x)++;
}

template <typename Ctx>
void run()
{
  int A[10] = {0};
  stream_ctx ctx;

  auto l = ctx.logical_data(A);

  for (size_t k = 0; k < 10; k++)
  {
    // size_t h = l.hash();
    // fprintf(stderr, "iter %zu : logical data hash %zu ctx.hash %zu\n", k, h, ctx.hash());

    ctx.task(l.rw())->*[](cudaStream_t stream, auto sA) {
      inc_kernel<<<1, 10, 0, stream>>>(sA);
    };
  }

  ctx.host_launch(l.read())->*[&](auto /*unused*/) {
    // fprintf(stderr, "HOST end : logical data hash %zu ctx.hash %zu\n", l.hash(), ctx.hash());
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
