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

// UNSUPPORTED: pre-sm-70

// <cuda/barrier>

#include <cuda/barrier>

int main(int, char**){
    NV_IF_TARGET(
        NV_IS_DEVICE, (
            __shared__ cuda::barrier<cuda::thread_scope_block> bar;
            if (threadIdx.x == 0) {
                init(&bar, blockDim.x);
            }
            __syncthreads();

            // barrier_arrive_tx should fail on SM70 and SM80, because it is hidden.
            auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);

#if defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__
            static_assert(false, "Fail manually for SM90 and up.");
#endif // __CUDA_MINIMUM_ARCH__
    ));
    return 0;
}
