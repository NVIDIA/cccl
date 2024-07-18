//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "test_macros.h"
#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4505)
#endif

#include <cuda/annotated_ptr>
#include <cuda/std/cassert>

#if defined(DEBUG)
#  define DPRINTF(...)     \
    {                      \
      printf(__VA_ARGS__); \
    }
#else
#  define DPRINTF(...) \
    do                 \
    {                  \
    } while (false)
#endif

__device__ __host__ void assert_rt_wrap(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess)
  {
#ifndef TEST_COMPILER_NVRTC
    printf("assert: %s %s %d\n", cudaGetErrorString(code), file, line);
#endif
    assert(code == cudaSuccess);
  }
}
#define assert_rt(ret)                         \
  {                                            \
    assert_rt_wrap((ret), __FILE__, __LINE__); \
  }

template <typename T, int N>
__device__ __host__ __noinline__ T* alloc(bool shared = false)
{
  T* arr = nullptr;

  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (
      if (!shared) { arr = (T*) malloc(N * sizeof(T)); } else {
        __shared__ T data[N];
        arr = data;
      }),
    assert_rt(cudaMallocManaged((void**) &arr, N * sizeof(T)));)

  for (int i = 0; i < N; ++i)
  {
    arr[i] = i;
  }
  return arr;
}

template <typename T>
__device__ __host__ __noinline__ void dealloc(T* arr, bool shared)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (if (!shared) free(arr);), assert_rt(cudaFree(arr));)
}
