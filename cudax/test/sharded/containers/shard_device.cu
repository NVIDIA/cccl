//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief `shard<T>` as a kernel argument: a shard may be passed by value to a
 *        `__global__` function and used there through its host+device
 *        accessors -- `begin()`/`end()`, `size_bytes()`, `empty()`,
 *        `contains()` and the index conversions.
 *
 * Only the span members (`data`, `size`, `global_offset`) are meaningful in
 * device code. `place`, `exec` and `stream` are host-side placement metadata:
 * they ride along in the parameter but must not be touched from the device.
 * This test pins that boundary -- it is what keeps a caller from having to
 * unpack a shard into loose pointer/size arguments at every launch.
 */

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
// Writes each element's GLOBAL index, reached through the shard's own
// iterators and index conversion -- no loose offset argument.
__global__ void fill_global_index_kernel(shard<unsigned long long> s)
{
  const size_t tid    = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (auto it = s.begin() + tid; it < s.end(); it += stride)
  {
    const size_t local = static_cast<size_t>(it - s.begin());
    *it                = static_cast<unsigned long long>(s.to_global(local));
  }
}

// Same traversal through a const shard, exercising the const overloads and
// the remaining host+device accessors.
__global__ void check_kernel(const shard<unsigned long long> s, int* error)
{
  const size_t tid    = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  if (tid == 0)
  {
    if (s.empty() || s.size_bytes() != s.size * sizeof(unsigned long long))
    {
      atomicExch(error, 1);
    }
    if (!s.contains(s.global_begin()) || s.contains(s.global_end()))
    {
      atomicExch(error, 1);
    }
  }

  for (const auto* it = s.begin() + tid; it < s.end(); it += stride)
  {
    const size_t global = s.to_global(static_cast<size_t>(it - s.begin()));
    if (*it != static_cast<unsigned long long>(global))
    {
      atomicExch(error, 1);
    }
    if (!s.contains(global))
    {
      atomicExch(error, 1);
    }
  }
}

void test_shard_as_kernel_argument()
{
  auto group = place_group{make_locality_domain_grid(0)};
  // Odd size so shards are uneven and the last one is a partial block
  const size_t n = (1 << 20) + 4097;

  auto arr = sharded_array<unsigned long long>::allocate(group, n);
  EXPECT(arr.num_shards() >= 1UL);
  EXPECT(arr.size() == n);

  // Produce: the shard itself is the kernel argument
  arr.each_shard->*[](auto& s) {
    const int block = 256;
    const int grid  = static_cast<int>((s.size + block - 1) / block);
    fill_global_index_kernel<<<grid, block, 0, s.stream>>>(s);
    cuda_safe_call(cudaGetLastError());
  };
  arr.sync();

  // Verify on the host: every element holds its global index
  ::std::vector<unsigned long long> host(n);
  arr.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<unsigned long long>(i));
  }

  // Verify again from device code, through a const shard argument
  ::std::vector<int*> errors(arr.num_shards(), nullptr);
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    const auto& s = arr.shard(i);
    cuda_safe_call(cudaSetDevice(0));
    cuda_safe_call(cudaMalloc(&errors[i], sizeof(int)));
    cuda_safe_call(cudaMemset(errors[i], 0, sizeof(int)));

    const int block = 256;
    const int grid  = static_cast<int>((s.size + block - 1) / block);
    check_kernel<<<grid, block, 0, s.stream>>>(s, errors[i]);
    cuda_safe_call(cudaGetLastError());
  }
  arr.sync();

  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    int h_error = -1;
    cuda_safe_call(cudaMemcpy(&h_error, errors[i], sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT(h_error == 0);
    cuda_safe_call(cudaFree(errors[i]));
  }
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  test_shard_as_kernel_argument();

  return 0;
}
