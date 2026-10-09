// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: no_execute
// UNSUPPORTED: nvrtc
// ADDITIONAL_COMPILE_DEFINITIONS: NDEBUG

#ifndef CCCL_ENABLE_ASSERTIONS
#  error "Should be compiled with CCCL_ENABLE_ASSERTIONS"
#endif // !CCCL_ENABLE_ASSERTIONS

#include <cuda/std/cassert>

#include "test_macros.h"

TEST_FUNC inline bool failed_on_host()
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, return true;, return false;)
}

int main(int, char**)
{
  _CCCL_ASSERT(failed_on_host(), "Should fail on host even with NDEBUG");
  return 0;
}
