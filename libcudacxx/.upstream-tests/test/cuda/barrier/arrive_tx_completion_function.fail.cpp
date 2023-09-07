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
#include "test_macros.h"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

struct CF {
  __host__ __device__ CF() {}
  __device__ void operator()() const {
    // do nothing
  }
  int x = 1;
};

int main(int, char**){
    NV_IF_TARGET(
        NV_IS_DEVICE, (
            // Use completion function. This is not yet supported. Check that
            // static_assert fails.
            CF cf{};

            __shared__ cuda::barrier<cuda::thread_scope_block, decltype(cf)> bar;
            if (threadIdx.x == 0) {
                init(&bar, (int) blockDim.x, cf);
            }
            __syncthreads();

            // Should fail due to CF:
            auto token = cuda::device::arrive_tx(bar, 1, 0);
    ));
    return 0;
}
