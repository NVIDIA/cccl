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

#include <cuda/barrier>

#ifndef __cccl_lib_local_barrier_arrive_tx
static_assert(false, "should define __cccl_lib_local_barrier_arrive_tx");
#endif // __cccl_lib_local_barrier_arrive_tx

int main(int, char**)
{
  return 0;
}
