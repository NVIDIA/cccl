//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_calloc_success(cuda::std::size_t n)
{
  T* ptr = static_cast<T*>(cuda::std::calloc(n, sizeof(T)));

  // check that the memory was allocated
  assert(ptr != nullptr);

  // check that the memory is zeroed
  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    assert(ptr[i] == T{});
  }

  // check memory alignment
  assert(((alignof(T) - 1) & reinterpret_cast<cuda::std::uintptr_t>(ptr)) == 0);

  cuda::std::free(ptr);
}

template <class T>
__host__ __device__ void test_calloc_fail(cuda::std::size_t n)
{
  T* ptr = static_cast<T*>(cuda::std::calloc(n, sizeof(T)));

  // check that the memory was not allocated
  assert(ptr == nullptr);
}

struct BigStruct
{
  static constexpr cuda::std::size_t n = 32;

  int data[n];

  __host__ __device__ bool operator==(const BigStruct& other) const
  {
    for (cuda::std::size_t i{}; i < n; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }

    return true;
  }
};

struct alignas(cuda::std::max_align_t) AlignedStruct
{
  static constexpr cuda::std::size_t n = 32;

  char data[n];

  __host__ __device__ bool operator==(const AlignedStruct& other) const
  {
    for (cuda::std::size_t i{}; i < n; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }

    return true;
  }
};

__host__ __device__ void test()
{
  test_calloc_success<int>(10);
  test_calloc_success<char>(128);
  test_calloc_success<double>(8);
  test_calloc_success<BigStruct>(4);
  test_calloc_success<AlignedStruct>(16);

  test_calloc_fail<int>(cuda::std::numeric_limits<cuda::std::size_t>::max());
}

int main(int, char**)
{
  test();

  return 0;
}
