/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This is a simple example demonstrating the use of CCCL functionality from Thrust, CUB, and libcu++.

The example computes the sum of an array of integers using a simple parallel reduction. Each thread block
computes the sum of a subset of the array using cuB::BlockRecuce. The sum of each block is then reduced
to a single value using an atomic add via cuda::atomic_ref from libcu++. The result is stored in a device_vector
from Thrust. The sum is then printed to the console.
*/

#include <cub/block/block_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/atomic>

#include <cstdio>

constexpr int block_size = 256;

__global__ void sumKernel(int const* data, int* result, std::size_t N)
{
  using BlockReduce = cub::BlockReduce<int, block_size>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int sum = 0;
  if (index < N)
  {
    sum += data[index];
  }

  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0)
  {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*result);
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}

int main()
{
  std::size_t N = 1000;
  thrust::device_vector<int> data(N, 1);
  thrust::device_vector<int> result(1);

  int num_blocks = (N + block_size - 1) / block_size;

  sumKernel<<<num_blocks, block_size>>>(
    thrust::raw_pointer_cast(data.data()), thrust::raw_pointer_cast(result.data()), N);

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  std::cout << "Sum: " << result[0] << std::endl;

  assert(result[0] == N);

  return 0;
}
