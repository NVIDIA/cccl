// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc

// The release-mode declaration must also match an existing system declaration.
#undef NDEBUG
#include <cassert>
#define NDEBUG

#include <cuda/std/cassert>

int main(int, char**)
{
  _CCCL_ASSERT(true, "Should succeed on host");
  _CCCL_VERIFY(true, "Should succeed on host");
  return 0;
}
