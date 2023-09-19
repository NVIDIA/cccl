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

// <cuda/barrier>

#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move
#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_async_tx is not available.");
#endif // __CUDA_MINIMUM_ARCH__

__device__ alignas(16) int gmem_x[2048];


int main(int, char**)
{
    NV_IF_TARGET(NV_IS_HOST,(
        //Required by concurrent_agents_launch to know how many we're launching
        cuda_thread_count = 512;
    ));

    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            using barrier_t = cuda::barrier<cuda::thread_scope_block>;
            __shared__ alignas(16) int smem_x[1024];
            __shared__ barrier_t bar;
            if (threadIdx.x == 0) {
                init(&bar, blockDim.x);
            }

            barrier_t::arrival_token token;
            if (threadIdx.x == 0) {
                cuda::device::memcpy_async_tx(smem_x, gmem_x, cuda::aligned_size_t<16>(sizeof(smem_x)), bar);
                token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_x));
            } else {
                auto token = bar.arrive(1);
            }
            bar.wait(cuda::std::move(token));

            // smem_x contains the contents of gmem_x[0], ..., gmem_x[1023]
            smem_x[threadIdx.x] += 1;
        )
    );
    return 0;
}
