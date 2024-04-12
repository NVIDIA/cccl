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
// UNSUPPORTED: no_execute

// <cuda/barrier>

#include <cuda/barrier>

#include "test_macros.h"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

__device__ uint64_t bar_storage;

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (cuda::barrier<cuda::thread_scope_block> * bar_ptr;
     bar_ptr = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(bar_storage);

     if (threadIdx.x == 0) { init(bar_ptr, blockDim.x); } __syncthreads();

     // Should fail because the barrier is in device memory.
     cuda::device::barrier_expect_tx(*bar_ptr, 1);));
  return 0;
}
