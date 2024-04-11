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

template <typename T, typename P>
__device__ __host__ __noinline__ void test(P ap, bool shared = false)
{
  T* arr = alloc<T, ARR_SZ>(shared);

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }

  dealloc<T>(arr, shared);
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>(cuda::access_property::normal{});
  test<int>(cuda::access_property::persisting{});
  test<int>(cuda::access_property::streaming{});
  test<int>(cuda::access_property::global{});
  test<int>(cuda::access_property{});
  test<int>(cuda::access_property::shared{}, true);
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
