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

// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert_kernel(hashtable h, const reserved::KeyValue* kvs, unsigned int numkvs)
{
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadid < numkvs)
  {
    h.insert(kvs[threadid]);
  }
}

void gpu_hashtable_insert(hashtable d_h, const reserved::KeyValue* device_kvs, unsigned int num_kvs, cudaStream_t stream)
{
  // Have CUDA calculate the thread block size
  auto [mingridsize, threadblocksize] = reserved::compute_occupancy(gpu_hashtable_insert_kernel);

  // Insert all the keys into the hash table
  int gridsize = ((uint32_t) num_kvs + threadblocksize - 1) / threadblocksize;
  gpu_hashtable_insert_kernel<<<gridsize, threadblocksize, 0, stream>>>(d_h, device_kvs, (uint32_t) num_kvs);
}

int main()
{
  stream_ctx ctx;

  // This constructor automatically initializes an empty hashtable on the host
  hashtable h;
  auto lh = ctx.logical_data(h);

  // Create an array of values on the host
  reserved::KeyValue kvs_array[16];
  for (uint32_t i = 0; i < 16; i++)
  {
    kvs_array[i].key   = i * 10;
    kvs_array[i].value = 17 + i * 14;
  }
  auto h_kvs_array = ctx.logical_data(make_slice(&kvs_array[0], 16));

  ctx.task(lh.rw(), h_kvs_array.read())->*[](auto stream, auto h, auto a) {
    gpu_hashtable_insert(h, a.data_handle(), static_cast<uint32_t>(a.extent(0)), stream);
  };

  ctx.finalize();

  // Thanks to the write-back mechanism on lh, h has been updated
  for (size_t i = 0; i < 16; i++)
  {
    assert(h.get(i * 10) == 17 + i * 14);
  }
}
