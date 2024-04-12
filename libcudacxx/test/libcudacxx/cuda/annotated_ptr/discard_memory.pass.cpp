//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: c++98, c++03

#include "utils.h"
#define ARR_SZ 128

template <typename T>
__device__ __host__ __noinline__ void test(bool shared = false)
{
  T* arr = alloc<T, ARR_SZ>(shared);

  cuda::discard_memory(arr, ARR_SZ);

  dealloc<T>(arr, shared);
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>();
}

__global__ void test_kernel()
{
  test_all();
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}
