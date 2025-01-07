//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: clang && !nvcc
// UNSUPPORTED: c++98, c++03, c++11, c++14

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

#include "manual/shfl_test.h"

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, cuda_thread_count = 32;)
  test_shfl_full_mask();
  test_shfl_partial_mask();
  test_shfl_partial_warp();
  return 0;
}
