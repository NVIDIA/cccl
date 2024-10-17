//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// We compile with CCCL_ENABLE_ASSERTIONS, but want to enable only host assertions
#undef CCCL_ENABLE_ASSERTIONS
#define CCCL_ENABLE_HOST_ASSERTIONS

#include <cuda/std/cassert>

__host__ __device__ inline bool failed_on_device()
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, return false;, return true;)
}

int main(int, char**)
{
  _CCCL_ASSERT(failed_on_device(), "Should fail on device");
  return 0;
}
