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
 * @brief `allocate_contiguous` contract tests: one VA range with per-place
 *        physical backing, exact logical shard boundaries, and the
 *        read-as-one-array guarantee — per-shard writes must be visible to a
 *        single unmodified kernel spanning the whole range through
 *        `contiguous_data()`, and vice versa.
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdio>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::cuda_try;
using cuda::experimental::places::place_group;

namespace
{
// Unmodified downstream consumer: reads the WHOLE array through one base
// pointer, with no knowledge of shards
__global__ void check_whole_kernel(const long long* base, size_t n, int* error)
{
  size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (tid < n)
  {
    if (base[tid] != static_cast<long long>(tid) + 1)
    {
      atomicExch(error, 1);
    }
  }
}

// Unmodified producer: writes the WHOLE array through one base pointer
__global__ void write_whole_kernel(long long* base, size_t n)
{
  size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (tid < n)
  {
    base[tid] = 2 * static_cast<long long>(tid);
  }
}

// Per-shard producer, launched shard by shard on the shard's place
__global__ void write_shard_kernel(long long* data, size_t n, size_t global_offset)
{
  size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (tid < n)
  {
    data[tid] = static_cast<long long>(global_offset + tid) + 1;
  }
}

void test_layout_contract()
{
  auto group = place_group::by_locality_domains({0});
  // Odd size so shards are uneven and boundaries are not granule-aligned
  const size_t n = (1 << 21) + 12345;

  auto arr = sharded_array<long long>::allocate_contiguous(group, n);
  EXPECT(arr.is_contiguous());
  EXPECT(arr.contiguous_data() != nullptr);
  EXPECT(arr.size() == n);
  EXPECT(arr.is_owning()); // the array owns the memory, via its VMM backing
  EXPECT(arr.get_ownership() == ownership::owning_backing);
  EXPECT(arr.validate());

  // Logical shard boundaries are EXACT: shard(i).data == base + global_offset
  long long* base = arr.contiguous_data();
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    EXPECT(arr.shard(i).data == base + arr.shard(i).global_offset);
  }

  // Dense: no inter-shard padding
  size_t covered = 0;
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    EXPECT(arr.shard(i).global_offset == covered);
    covered += arr.shard(i).size;
  }
  EXPECT(covered == n);

  // Per-shard placement correspondence: shard i carries place i of the group
  EXPECT(arr.num_shards() == group.size());
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    EXPECT(arr.shard(i).place == group.place(i).affine_data_place());
    EXPECT(arr.shard(i).exec == group.place(i));
  }

  // A non-contiguous array reports no base pointer
  auto plain = sharded_array<long long>::allocate(group, 1000);
  EXPECT(!plain.is_contiguous());
  EXPECT(plain.contiguous_data() == nullptr);
}

void test_whole_kernel_visibility()
{
  auto group     = place_group::by_locality_domains({0});
  const size_t n = (1 << 21) + 999;

  auto arr        = sharded_array<long long>::allocate_contiguous(group, n);
  long long* base = arr.contiguous_data();

  // Produce per shard, on each shard's own place and stream
  arr.each_shard->*[](auto& s) {
    const int block = 256;
    const int grid  = static_cast<int>((s.size + block - 1) / block);
    write_shard_kernel<<<grid, block, 0, s.stream>>>(s.data, s.size, s.global_offset);
    cuda_safe_call(cudaGetLastError());
  };
  arr.sync();

  // Consume with ONE kernel spanning the whole range through the base pointer
  cuda_safe_call(cudaSetDevice(0));
  int* d_error = nullptr;
  cuda_safe_call(cudaMalloc(&d_error, sizeof(int)));
  cuda_safe_call(cudaMemset(d_error, 0, sizeof(int)));

  const int block = 256;
  const int grid  = static_cast<int>((n + block - 1) / block);
  check_whole_kernel<<<grid, block>>>(base, n, d_error);
  cuda_safe_call(cudaGetLastError());

  int h_error = -1;
  cuda_safe_call(cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT(h_error == 0);

  // And the reverse: one whole-array producer, per-shard consumers
  write_whole_kernel<<<grid, block>>>(base, n);
  cuda_safe_call(cudaGetLastError());
  cuda_safe_call(cudaDeviceSynchronize());

  ::std::vector<long long> host(n);
  arr.copy_to_host(host.data());
  for (size_t i = 0; i < n; i += 4097) // sampled check
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i));
  }
  EXPECT(host[n - 1] == 2 * static_cast<long long>(n - 1));

  cuda_safe_call(cudaFree(d_error));
}

void test_non_affine_spec_refused()
{
  // The contiguous backing places physical blocks at each spec's exec
  // place's affine data place; a spec naming any other data_place must
  // throw rather than be silently ignored.
  auto group = place_group::by_locality_domains({0});
  auto place = group.place(0);
  bool threw = false;
  try
  {
    ::std::vector<shard_spec> specs;
    specs.emplace_back(1024, cuda::experimental::places::data_place::host(), place, group.get_stream(0));
    auto bad = sharded_array<long long>::allocate_contiguous(specs);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}

void test_empty_and_cleanup()
{
  auto empty = sharded_array<long long>::allocate_contiguous(::std::vector<shard_spec>{});
  EXPECT(!empty.is_contiguous());
  EXPECT(empty.size() == 0UL);

  // clear() releases the VMM backing
  auto group = place_group::by_locality_domains({0});
  auto arr   = sharded_array<long long>::allocate_contiguous(group, 1 << 20);
  EXPECT(arr.is_contiguous());
  arr.clear();
  EXPECT(!arr.is_contiguous());
  EXPECT(arr.size() == 0UL);
}
} // namespace

int main()
{
  cuda_try(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  if (!contiguous_backing_supported())
  {
    printf("VMM not supported on this device, skipping tests.\n");
    return 0;
  }

  test_layout_contract();
  test_whole_kernel_visibility();
  test_non_affine_spec_refused();
  test_empty_and_cleanup();

  return 0;
}
