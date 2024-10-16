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

template <typename S1, typename S2>
__global__ void copy_kernel(S1 sA, S2 sB)
{
  sB(threadIdx.x) = sA(threadIdx.x);
}

template <typename Ctx>
void run()
{
  int A[10] = {0};
  int B[10] = {0};
  Ctx ctx;

  auto lA = ctx.logical_data(A);
  auto lB = ctx.logical_data(B);

  for (size_t k = 0; k < 10; k++)
  {
    // fprintf(stderr, "iter %zu - 0 : ctx.hash %zu\n", k, ctx.hash());
    ctx.task(lA.rw())->*[](cudaStream_t stream, auto sA) {
      inc_kernel<<<1, 10, 0, stream>>>(sA);
    };

    // fprintf(stderr, "iter %zu - 1 : ctx.hash %zu\n", k, ctx.hash());
    ctx.task(lA.read(), lB.rw())->*[](cudaStream_t stream, auto sA, auto sB) {
      copy_kernel<<<1, 10, 0, stream>>>(sA, sB);
    };

    // fprintf(stderr, "iter %zu - 2 : ctx.hash %zu\n", k, ctx.hash());
    ctx.task(lB.rw())->*[](cudaStream_t stream, auto sB) {
      inc_kernel<<<1, 10, 0, stream>>>(sB);
    };

    // fprintf(stderr, "iter %zu - 3 : ctx.hash %zu\n", k, ctx.hash());
  }

  ctx.host_launch(lA.read(), lB.read())->*[&](auto /*unused*/, auto /*unused*/) {
    // fprintf(stderr, "HOST END : ctx.hash %zu\n", ctx.hash());
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
