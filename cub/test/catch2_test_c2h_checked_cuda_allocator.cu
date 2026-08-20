// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/detail/execution_policy.h>
#include <thrust/execution_policy.h>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/stream>

#include <algorithm>
#include <new> // std::bad_alloc

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/checked_allocator.cuh>

std::size_t get_alloc_bytes()
{
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  REQUIRE(status == cudaSuccess);

  // Find a size that's > free but < total, preferring to return more than total if the values are
  // too close.
  constexpr std::size_t one_MiB = 1024 * 1024;
  const std::size_t alloc_bytes = ::std::max(total_bytes - one_MiB, free_bytes + one_MiB);
  CAPTURE(free_bytes, total_bytes, alloc_bytes);
  return alloc_bytes;
}

CUB_TEST("c2h::device_vector throws when requested allocations exceed free device memory",
         "[c2h][checked_cuda_allocator][device_vector]",
         CUB_SMALL)
{
  c2h::device_vector<char> vec;

  const std::size_t alloc_bytes = get_alloc_bytes();
  REQUIRE_THROWS_AS(vec.resize(alloc_bytes), std::bad_alloc);
}

CUB_TEST("c2h::device_policy throws when requested allocations exceed free device memory",
         "[c2h][checked_cuda_allocator][device_policy]",
         CUB_SMALL)
{
  cuda::std::pair<char*, std::ptrdiff_t> buffer{nullptr, 0};
  auto policy = thrust::detail::derived_cast(thrust::detail::strip_const(c2h::device_policy));

  const std::size_t alloc_bytes = get_alloc_bytes();
  REQUIRE_THROWS_AS(
    buffer = thrust::detail::get_temporary_buffer<char>(policy, static_cast<std::ptrdiff_t>(alloc_bytes)),
    std::bad_alloc);

  thrust::detail::return_temporary_buffer(policy, buffer.first, buffer.second);
}

CUB_TEST("c2h::checked_device_memory_resource throws when requested allocations exceed free device memory",
         "[c2h][checked_cuda_allocator][device_buffer]",
         CUB_SMALL)
{
  STATIC_REQUIRE(cuda::mr::synchronous_resource_with<c2h::checked_device_memory_resource, cuda::mr::device_accessible>);

  int current_device = 0;
  REQUIRE(cudaSuccess == cudaGetDevice(&current_device));

  const auto stream      = cuda::stream_ref{cudaStream_t{}};
  const auto alloc_bytes = get_alloc_bytes();
  REQUIRE_THROWS_AS(c2h::make_device_buffer<char>(stream, cuda::device_ref{current_device}, alloc_bytes, cuda::no_init),
                    std::bad_alloc);

  const auto small_size = std::size_t{1024};
  const auto small = c2h::make_device_buffer<char>(stream, cuda::device_ref{current_device}, small_size, cuda::no_init);
  REQUIRE(small.size() == small_size);
  REQUIRE(small.data() != nullptr);

  const auto empty =
    c2h::make_device_buffer<char>(stream, cuda::device_ref{current_device}, std::size_t{0}, cuda::no_init);
  REQUIRE(empty.size() == 0);
  REQUIRE(empty.data() == nullptr);

  auto resource                      = c2h::checked_device_memory_resource{cuda::device_ref{current_device}};
  constexpr auto invalid_alignment   = ::cuda::mr::default_cuda_malloc_alignment - 1;
  constexpr auto invalid_alloc_bytes = std::size_t{1};
  REQUIRE_THROWS_AS(resource.allocate_sync(invalid_alloc_bytes, invalid_alignment), std::bad_alloc);
}
