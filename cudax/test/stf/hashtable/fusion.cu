//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/interfaces/hashtable_linearprobing.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

// Iterate over every item in the hashtableA, and add them to B

__global__ void gpu_merge_hashtable(hashtable A, const hashtable B)
{
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  while (threadid < reserved::kHashTableCapacity)
  {
    if (B.addr[threadid].key != reserved::kEmpty)
    {
      uint32_t value = B.addr[threadid].value;
      if (value != reserved::kEmpty)
      {
        //    printf("INSERTING key %d value %d\n", pHashTableB[threadid].key, value);
        A.insert(B.addr[threadid]);
      }
    }
    threadid += blockDim.x * gridDim.x;
  }
}

int main()
{
  stream_ctx ctx;

  hashtable A;
  A.insert(reserved::KeyValue(107, 4));
  A.insert(reserved::KeyValue(108, 6));

  hashtable B;
  B.insert(reserved::KeyValue(7, 14));
  B.insert(reserved::KeyValue(8, 16));

  auto lA = ctx.logical_data(A);
  auto lB = ctx.logical_data(B);

  ctx.task(lA.rw(), lB.read())->*[](auto stream, auto hA, auto hB) {
    gpu_merge_hashtable<<<32, 128, 0, stream>>>(hA, hB);
    cuda_safe_call(cudaGetLastError());
  };

  ctx.host_launch(lA.read())->*[](auto hA) {
    EXPECT(hA.get(107) == 4);
    EXPECT(hA.get(108) == 6);
    EXPECT(hA.get(7) == 14);
    EXPECT(hA.get(8) == 16);
  };

  ctx.finalize();
}
