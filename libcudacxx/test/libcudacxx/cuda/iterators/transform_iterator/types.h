//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_TRANSFORM_ITERATOR_TYPES_H
#define TEST_CUDA_TRANSFORM_ITERATOR_TYPES_H

#include <cuda/std/utility>

#include "test_macros.h"

struct TimesTwo
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x * 2;
  }
};

struct PlusOneMutable
{
  __host__ __device__ constexpr int operator()(int x)
  {
    return x + 1;
  }
};

struct PlusOne
{
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

struct Increment
{
  __host__ __device__ constexpr int& operator()(int& x)
  {
    return ++x;
  }
};

struct IncrementRvalueRef
{
  __host__ __device__ constexpr int&& operator()(int& x)
  {
    return cuda::std::move(++x);
  }
};

struct PlusOneNoexcept
{
  __host__ __device__ constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

struct PlusWithMutableMember
{
  int val_ = 0;
  __host__ __device__ constexpr PlusWithMutableMember(const int val) noexcept
      : val_(val)
  {}
  __host__ __device__ constexpr int operator()(int x) noexcept
  {
    return x + val_++;
  }
};

#endif // TEST_CUDA_TRANSFORM_ITERATOR_TYPES_H
