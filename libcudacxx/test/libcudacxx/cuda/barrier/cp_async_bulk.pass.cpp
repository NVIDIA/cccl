//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90
// UNSUPPORTED: nvcc-11

// <cuda/barrier>

#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move

#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

static constexpr int buf_len = 1024;
__device__ int gmem_buffer[buf_len];

__device__ void test()
{
  // SETUP: fill global memory buffer
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
  {
    gmem_buffer[i] = i;
  }
  // Ensure that writes to global memory are visible to others, including
  // those in the async proxy.
  __threadfence();
  __syncthreads();

  // TEST: Add i to buffer[i]
  alignas(16) __shared__ int smem_buffer[buf_len];
  __shared__ barrier* bar;
  if (threadIdx.x == 0)
  {
    init(bar, blockDim.x);
  }
  __syncthreads();

  // Load data:
  uint64_t token;
  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_global_to_shared(smem_buffer, gmem_buffer, sizeof(smem_buffer), *bar);
    token = cuda::device::barrier_arrive_tx(*bar, 1, sizeof(smem_buffer));
  }
  else
  {
    token = bar->arrive();
  }
  bar->wait(cuda::std::move(token));

  // Update in shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
  {
    smem_buffer[i] += i;
  }
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  // Write back to global memory:
  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_shared_to_global(gmem_buffer, smem_buffer, sizeof(smem_buffer));
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  __threadfence();
  __syncthreads();

  // TEAR-DOWN: check that global memory is correct
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
  {
    assert(gmem_buffer[i] == 2 * i);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST,
               (
                 // Required by concurrent_agents_launch to know how many we're launching
                 cuda_thread_count = 512;));

  NV_DISPATCH_TARGET(NV_IS_DEVICE, (test();));
  return 0;
}
