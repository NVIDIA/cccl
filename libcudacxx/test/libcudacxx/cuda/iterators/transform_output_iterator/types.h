//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
#define TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"

struct PlusOne
{
  __host__ __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};

struct PlusOneMutable
{
  __host__ __device__ constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

struct PlusOneMayThrow
{
  __host__ __device__ constexpr int operator()(int x)
  {
    return x + 1;
  }
};

#if !TEST_COMPILER(NVRTC)
struct PlusOneHost
{
  constexpr PlusOneHost() noexcept {}
  constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // !TEST_COMPILER(NVRTC)

#if TEST_HAS_CUDA_COMPILER()
struct PlusOneDevice
{
  __device__ constexpr PlusOneDevice() noexcept {}
  __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // TEST_HAS_CUDA_COMPILER()

struct NotDefaultConstructiblePlusOne
{
  __host__ __device__ constexpr NotDefaultConstructiblePlusOne(int) noexcept {}
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

#endif // TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
