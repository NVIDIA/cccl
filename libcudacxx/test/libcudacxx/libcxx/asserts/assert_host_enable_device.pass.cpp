//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// We compile with CCCL_ENABLE_ASSERTIONS, but want to enable only device assertions
#undef CCCL_ENABLE_ASSERTIONS
#define CCCL_ENABLE_DEVICE_ASSERTIONS

#include <cuda/std/cassert>

int main(int, char**)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (_CCCL_ASSERT(true, "Should not fail on device");), (_CCCL_ASSERT(false, "Should not fail on host");))
  return 0;
}
