//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Verify that graph_ctx tasks whose captured child graphs contain
 *        memory allocation/free nodes (from cudaMallocAsync) work correctly.
 *
 * Before the move-ownership fix, cudaGraphAddChildGraphNode (clone semantics)
 * rejected such child graphs with CUDA_ERROR_NOT_SUPPORTED.  CTK 13+ exposes
 * cudaGraphChildGraphOwnershipMove via cudaGraphAddNode which transfers the
 * child graph to the parent instead of cloning it.
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void fill_kernel(int* ptr, int n, int val)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    ptr[tid] = val;
  }
}

int main()
{
#if _CCCL_CTK_BELOW(13, 0)
  fprintf(stderr, "Waiving test: cudaGraphChildGraphOwnershipMove requires CTK 13+.\n");
#else
  constexpr int N = 256;
  int host_data[N];
  for (int i = 0; i < N; i++)
  {
    host_data[i] = 0;
  }

  graph_ctx ctx;
  auto ldata = ctx.logical_data(host_data);

  // The lambda receives cudaStream_t, so graph_ctx uses stream capture.
  // cudaMallocAsync/cudaFreeAsync on that stream produce mem-alloc/free
  // graph nodes inside the captured child graph.
  ctx.task(ldata.rw())->*[](cudaStream_t s, auto sdata) {
    int* tmp = nullptr;
    cuda_safe_call(cudaMallocAsync(&tmp, N * sizeof(int), s));
    fill_kernel<<<(N + 255) / 256, 256, 0, s>>>(tmp, N, 42);
    cuda_safe_call(cudaMemcpyAsync(sdata.data_handle(), tmp, N * sizeof(int), cudaMemcpyDeviceToDevice, s));
    cuda_safe_call(cudaFreeAsync(tmp, s));
  };

  ctx.finalize();

  for (int i = 0; i < N; i++)
  {
    assert(host_data[i] == 42);
  }
#endif // !_CCCL_CTK_BELOW(13, 0)
}
