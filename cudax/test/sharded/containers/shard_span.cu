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
 * @brief `shard<T>::span()` is what crosses into a kernel. A shard is a
 *        host-side handle (it carries places, which are host concepts); its
 *        span is the placeless view of the elements, trivially copyable, and
 *        is taken by value as a `__global__` parameter. `global_offset`
 *        travels alongside when the kernel needs the logical index space.
 */

#include <cuda/experimental/sharded.cuh>

#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
static_assert(::cuda::std::is_trivially_copyable_v<::cuda::std::span<unsigned long long>>);
static_assert(::cuda::std::is_trivially_copyable_v<::cuda::std::span<const unsigned long long>>);

// Writes each element's GLOBAL index through the span
__global__ void fill_global_index_kernel(::cuda::std::span<unsigned long long> s, size_t global_offset)
{
  const size_t tid    = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = tid; i < s.size(); i += stride)
  {
    s[i] = static_cast<unsigned long long>(global_offset + i);
  }
}

// Re-verifies from device code through the const span
__global__ void check_kernel(::cuda::std::span<const unsigned long long> s, size_t global_offset, int* error)
{
  const size_t tid    = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  for (size_t i = tid; i < s.size(); i += stride)
  {
    if (s[i] != static_cast<unsigned long long>(global_offset + i))
    {
      atomicExch(error, 1);
    }
  }
}

void test_span_as_kernel_argument()
{
  auto group = place_group{make_locality_domain_grid(0)};
  // Odd size so shards are uneven and the last block is partial
  const size_t n = (1 << 20) + 4097;

  auto arr = sharded_array<unsigned long long>::allocate(group, n);
  EXPECT(arr.num_shards() >= 1UL);
  EXPECT(arr.size() == n);

  // span() covers exactly the valid elements
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    const auto& s = arr.shard(i);
    EXPECT(s.span().data() == s.data);
    EXPECT(s.span().size() == s.size);
  }

  // Produce: the span is the kernel argument
  arr.each_shard->*[](auto& s) {
    const int block = 256;
    const int grid  = static_cast<int>((s.size + block - 1) / block);
    fill_global_index_kernel<<<grid, block, 0, s.stream>>>(s.span(), s.global_offset);
    cuda_safe_call(cudaGetLastError());
  };

  // Verify on the host. No sync needed: copy_to_host copies each shard on
  // s.stream, ordered after the kernel above, and is itself synchronous.
  ::std::vector<unsigned long long> host(n);
  arr.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<unsigned long long>(i));
  }

  // Verify from device code through the const span
  ::std::vector<int*> errors(arr.num_shards(), nullptr);
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    const auto& s = arr.shard(i);
    cuda_safe_call(cudaSetDevice(0));
    cuda_safe_call(cudaMalloc(&errors[i], sizeof(int)));
    cuda_safe_call(cudaMemset(errors[i], 0, sizeof(int)));

    const int block = 256;
    const int grid  = static_cast<int>((s.size + block - 1) / block);
    check_kernel<<<grid, block, 0, s.stream>>>(s.span(), s.global_offset, errors[i]);
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

void test_descriptor_span()
{
  // The portable descriptor offers the same view
  ::std::vector<unsigned long long> host(100);
  basic_shard_view<unsigned long long> d{host.data(), host.size(), 0, 0};
  EXPECT(d.span().data() == host.data());
  EXPECT(d.span().size() == host.size());
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  test_span_as_kernel_argument();
  test_descriptor_span();

  return 0;
}
