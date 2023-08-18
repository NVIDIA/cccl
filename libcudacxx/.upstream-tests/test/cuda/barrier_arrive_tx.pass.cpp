//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-90

#include <cuda/barrier>

#include "cuda_space_selector.h"
#include "test_macros.h"
#include <cstdint>              // uint and friends
#include <utility>              // std::move

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ bool run_arrive_tx_test(barrier &bar) {
  if (threadIdx.x == 0) {
      init(&bar, blockDim.x);
  }
  __syncthreads();

  auto token = bar.arrive_tx(1, 1);

  // Manually increase the transaction count of the barrier by blockDim.x.
  // This emulates a cp.async.bulk instruction or equivalently, a memcpy_async call.
  if (threadIdx.x == 0) {
      asm volatile(
          "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
          :
          : "r"((uint32_t) __cvta_generic_to_shared(cuda::device::barrier_native_handle(bar)))
          , "r"(blockDim.x)
          : "memory");
  }
  bar.wait(std::move(token));

  return true;
}



#ifdef __CUDACC_RTC__
__device__ void arrive_on_nvrtc()
{
    __shared__ barrier bar;
    assert(run_arrive_tx_test(bar));
}
#else

__global__ void arrive_on_kernel(bool* success)
{
    __shared__ barrier bar;
    *success = run_arrive_tx_test(bar);
}

void arrive_on_launch(bool* success_dptr, volatile bool* success_hptr, unsigned block_size)
{
    *success_hptr = false;
    printf("arrive_on_kernel<<<%u, %2u>>> ", 1, block_size);
    arrive_on_kernel<<<1, block_size>>>(success_dptr);
    cudaError_t result;
    CUDA_CALL(result, cudaDeviceSynchronize());
    CUDA_CALL(result, cudaGetLastError());
    printf("%s\n", *success_hptr ? "[ OK ]" : "[FAIL]");
    assert(*success_hptr);
}

void arrive_on()
{
    volatile bool* success_hptr;
    bool* success_dptr;
    int lanes_per_warp;

    cudaError_t result;
    CUDA_CALL(result, cudaHostAlloc(&success_hptr, sizeof(*success_hptr), cudaHostAllocMapped));
    CUDA_CALL(result, cudaHostGetDevicePointer(&success_dptr, (void*)success_hptr, 0));
    CUDA_CALL(result, cudaDeviceGetAttribute(&lanes_per_warp, cudaDevAttrWarpSize, 0));

    // 1 Thread
    {
        const unsigned block_size = 1;
        arrive_on_launch(success_dptr, success_hptr, block_size);
    }

    // 1 Warp
    {
        const unsigned block_size = 1 * lanes_per_warp;
        arrive_on_launch(success_dptr, success_hptr, block_size);
    }

    // 1 CTA
    {
        const unsigned block_size = 2 * lanes_per_warp;
        arrive_on_launch(success_dptr, success_hptr, block_size);
    }

    CUDA_CALL(result, cudaFreeHost((void*)success_hptr));
}

#endif

int main(int argc, char ** argv)
{
  NV_IF_TARGET(NV_IS_HOST,
    arrive_on();
  )

#ifdef TEST_COMPILER_NVRTC
    int cuda_thread_count = 64;
    int cuda_block_shmem_size = 40000;

    arrive_on_nvrtc(cuda_block_shmem_size - sizeof(cuda::barrier<cuda::thread_scope_block>) - sizeof(char *));
#endif // TEST_COMPILER_NVRTC

    return 0;
}
