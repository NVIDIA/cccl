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
#include <cuda/experimental/__stf/stream/reduction.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void gpu_merge_hashtable(hashtable A, const hashtable B)
{
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  while (threadid < B.get_capacity())
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

void cpu_merge_hashtable(hashtable A, const hashtable B)
{
  for (unsigned int i = 0; i < B.get_capacity(); i++)
  {
    if (B.addr[i].key != reserved::kEmpty)
    {
      uint32_t value = B.addr[i].value;
      if (value != reserved::kEmpty)
      {
        //    printf("INSERTING key %d value %d\n", pHashTableB[threadid].key, value);
        A.insert(B.addr[i]);
      }
    }
  }
}

class hashtable_fusion_t : public stream_reduction_operator<hashtable>
{
  void op(const hashtable& in, hashtable& inout, const exec_place& e, cudaStream_t s) override
  {
    if (e.affine_data_place() == data_place::host)
    {
      cuda_safe_call(cudaStreamSynchronize(s)); // TODO use a callback
      cpu_merge_hashtable(inout, in);
    }
    else
    {
      gpu_merge_hashtable<<<32, 32, 0, s>>>(inout, in);
    }
  }

  void init_op(hashtable& /*unused*/, const exec_place& /*unused*/, cudaStream_t /*unused*/) override
  {
    // This init operator is a no-op because hashtables are already
    // initialized as empty tables
  }
};

// A kernel to fill the hashtable with some fictious values
__global__ void fill_table(size_t dev_id, size_t cnt, hashtable h)
{
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int nthreads = blockDim.x * gridDim.x;

  for (unsigned int i = threadid; i < cnt; i += nthreads)
  {
    uint32_t key   = dev_id * 1000 + i;
    uint32_t value = 2 * i;

    reserved::KeyValue kvs(key, value);
    h.insert(kvs);
  }
}

int main()
{
  stream_ctx ctx;

  // Explicit capacity of 2048 entries
  hashtable refh(2048);
  auto h_handle = ctx.logical_data(refh);

  auto fusion_op = std::make_shared<hashtable_fusion_t>();

  for (size_t dev_id = 0; dev_id < 4; dev_id++)
  {
    ctx.task(h_handle.relaxed(fusion_op))->*[&](auto stream, auto h) {
      EXPECT(h.get_capacity() == 2048);
      fill_table<<<32, 32, 0, stream>>>(dev_id, 10, h);
    };
  }

  ctx.host_launch(h_handle.read())->*[&](auto h) {
    // Check that the table contains all values
    for (size_t dev_id = 0; dev_id < 4; dev_id++)
    {
      for (unsigned i = 0; i < 10; i++)
      {
        uint32_t key   = static_cast<uint32_t>(dev_id * 1000 + i);
        uint32_t value = 2 * i;

        EXPECT(h.get(key) == value);
      }
    }
  };

  ctx.finalize();
}
