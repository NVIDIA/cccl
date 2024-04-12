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

// <cuda/barrier>

#include "arrive_tx.h"

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (
      // Required by concurrent_agents_launch to know how many we're
      // launching. This can only be an int, because the nvrtc tests use grep
      // to figure out how many threads to launch.
      cuda_thread_count = 2;),
    NV_IS_DEVICE,
    (constexpr bool split_arrive_and_expect = true; test<split_arrive_and_expect>();));

  return 0;
}
