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

constexpr size_t array_size = 128;

template <typename T, typename P>
__device__ __host__ __noinline__ void test(P ap)
{
  T* arr = global_alloc<T, array_size>();

  cuda::apply_access_property(arr, array_size * sizeof(T), ap);

  for (size_t i = 0; i < array_size; ++i)
  {
    assert(arr[i] == i);
  }

  dealloc<T>(arr);
}

template <typename T, typename P>
__device__ __host__ __noinline__ void test_aligned(P ap)
{
  T* arr = global_alloc<T, array_size>();

  cuda::apply_access_property(arr, cuda::aligned_size_t<sizeof(T)>(array_size * sizeof(T)), ap);

  for (size_t i = 0; i < array_size; ++i)
  {
    assert(arr[i] == i);
  }

  dealloc<T>(arr);
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>(cuda::access_property::normal{});
  test<int>(cuda::access_property::persisting{});
  test_aligned<int>(cuda::access_property::normal{});
  test_aligned<int>(cuda::access_property::persisting{});
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}
