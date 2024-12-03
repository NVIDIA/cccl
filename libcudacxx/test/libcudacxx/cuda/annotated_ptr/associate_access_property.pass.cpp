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
__device__ __host__ __noinline__ void test(P ap)
{
  T* arr = global_alloc<T, ARR_SZ>();

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }

  dealloc<T>(arr);
}

template <typename T, typename P>
__device__ __host__ __noinline__ void test_shared(P ap)
{
  T* arr = shared_alloc<T, ARR_SZ>();

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>(cuda::access_property::normal{});
  test<int>(cuda::access_property::persisting{});
  test<int>(cuda::access_property::streaming{});
  test<int>(cuda::access_property::global{});
  test<int>(cuda::access_property{});
  NV_IF_TARGET(NV_IS_DEVICE, (test_shared<int>(cuda::access_property::shared{});))
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}
