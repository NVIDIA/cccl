//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: nvcc-12, nvcc-13.0, nvcc-13.1, nvcc-13.2, nvcc-13.3
// UNSUPPORTED: pre-sm-90

#include <cuda/barrier>

#include "test_macros.h"

TEST_DEVICE_FUNC void test_block_scope(cuda::shared_memory_barrier<>& bar)
{
  bar.remote_arrive();
  bar.remote_arrive_tx(1, 0);
  bar.remote_expect_tx(0);
  bar.remote_complete_tx(0);
}

int main(int, char**)
{
  return 0;
}
