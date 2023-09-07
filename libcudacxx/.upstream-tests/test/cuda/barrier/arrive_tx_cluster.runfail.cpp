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

#include <cooperative_groups.h>
#include <cuda/barrier>
#include "test_macros.h"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

int main(int, char**){
    NV_DISPATCH_TARGET(
        NV_IS_HOST, (
            cuda_cluster_size = 2;
        ),
        NV_IS_DEVICE, (
            __shared__ cuda::barrier<cuda::thread_scope_block> bar;

            if (threadIdx.x == 0) {
                init(&bar, blockDim.x);
            }
            namespace cg = cooperative_groups;
            auto cluster = cg::this_cluster();

            cluster.sync();

            cuda::barrier<cuda::thread_scope_block> *remote_bar;
            remote_bar = cluster.map_shared_rank(&bar, cluster.block_rank() ^ 1);

            // Should fail because the barrier is in device memory.
            auto token = cuda::device::arrive_tx(*remote_bar, 1, 0);
    ));
    return 0;
}
