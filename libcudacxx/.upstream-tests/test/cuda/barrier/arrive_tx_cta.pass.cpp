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

#include "arrive_tx.h"


int main(int, char**)
{
    NV_IF_TARGET(NV_IS_HOST, (
        // Required by concurrent_agents_launch to know how many we're
        // launching. This can only be an int, because the nvrtc tests use grep
        // to figure out how many threads to launch.
        cuda_thread_count = 256;
    ));

    // Run test on both host and device
    test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
    test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();

    // Repeat for device and system scope
    test<cuda::barrier<cuda::thread_scope_device>, global_memory_selector>();
    test<cuda::barrier<cuda::thread_scope_system>, global_memory_selector>();

    return 0;
}
