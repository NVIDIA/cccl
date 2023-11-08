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
// UNSUPPORTED: c++11
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90
// UNSUPPORTED: nvrtc
// NVRTC_SKIP_KERNEL_RUN // This will have effect once PR 433 is merged (line above should be removed.)

// <cuda/barrier>

#include "cp_async_bulk_tensor_generic.h"

// Define the size of contiguous tensor in global and shared memory.
//
// Note that the first dimension is the one with stride 1. This one must be a
// multiple of 4 to ensure that each new dimension starts at a 16-byte aligned
// offset.
//
// We have a separate variable for host and device because a constexpr
// std::initializer_list cannot be shared between host and device as some of its
// member functions take a const reference, which is unsupported by nvcc.
           constexpr std::initializer_list<int> GMEM_DIMS    {256};
__device__ constexpr std::initializer_list<int> GMEM_DIMS_DEV{256};
           constexpr std::initializer_list<int> SMEM_DIMS    {32};
__device__ constexpr std::initializer_list<int> SMEM_DIMS_DEV{32};

__device__ constexpr std::initializer_list<int> TEST_SMEM_COORDS[] = {
    {0},
    {4},
    {8}
};

constexpr size_t gmem_len = tensor_len(GMEM_DIMS);
constexpr size_t smem_len = tensor_len(SMEM_DIMS);

__device__ int gmem_tensor[gmem_len];

int main(int, char**)
{
    NV_DISPATCH_TARGET(
        NV_IS_HOST, (
            //Required by concurrent_agents_launch to know how many we're launching
            cuda_thread_count = 512;
            init_tensor_map(gmem_tensor, GMEM_DIMS, SMEM_DIMS);
        ),
        NV_IS_DEVICE, (
            for (auto smem_coord : TEST_SMEM_COORDS) {
                test<smem_len>(smem_coord, SMEM_DIMS_DEV, GMEM_DIMS_DEV, gmem_tensor, gmem_len);
            }
        )
    );
    return 0;
}
