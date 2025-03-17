//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no_execute
// UNSUPPORTED: nvrtc

// We compile with CCCL_ENABLE_ASSERTIONS
#ifndef CCCL_ENABLE_ASSERTIONS
#  error "Should be compiled with CCCL_ENABLE_ASSERTIONS"
#endif // !CCCL_ENABLE_ASSERTIONS

#include <cuda/std/cassert>

__host__ __device__ inline bool failed_on_host()
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, return true;, return false;)
}

int main(int, char**)
{
  _CCCL_ASSERT(failed_on_host(), "Should fail on host");
  return 0;
}
