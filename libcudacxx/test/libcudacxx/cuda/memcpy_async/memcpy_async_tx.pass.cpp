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

#include "cuda_space_selector.h" // shared_memory_selector
#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_async_tx is not available.");
#endif // __CUDA_MINIMUM_ARCH__

__device__ alignas(16) int gmem_x[2048];

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST,
               (
                 // Required by concurrent_agents_launch to know how many we're launching
                 cuda_thread_count = 512;));

  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (
      using barrier_t = cuda::barrier<cuda::thread_scope_block>; alignas(16) __shared__ int smem_x[1024];

      shared_memory_selector<barrier_t, constructor_initializer> sel;
      barrier_t* b = sel.construct(blockDim.x);

      // Initialize gmem_x
      for (int i = threadIdx.x; i < 2048; i += blockDim.x) { gmem_x[i] = i; } __syncthreads();

      barrier_t::arrival_token token;
      if (threadIdx.x == 0) {
        auto fulfillment = cuda::device::memcpy_async_tx(smem_x, gmem_x, cuda::aligned_size_t<16>(sizeof(smem_x)), *b);
        assert(fulfillment == cuda::async_contract_fulfillment::async);
        token = cuda::device::barrier_arrive_tx(*b, 1, sizeof(smem_x));
      } else { token = b->arrive(1); } b->wait(cuda::std::move(token));

      // assert that smem_x contains the contents of gmem_x[0], ..., gmem_x[1023]
      for (int i = threadIdx.x; i < 1024; i += blockDim.x) { assert(smem_x[i] == i); }));
  return 0;
}
