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

struct basic_functor
{
  __host__ __device__ constexpr void operator()(const cuda::std::ptrdiff_t val, const int expected) const noexcept
  {
    assert(val == expected); // asserts that the assigned value matches the index
  }
};

struct not_default_constructible_functor
{
  __host__ __device__ not_default_constructible_functor(const int) {}
  __host__ __device__ constexpr void operator()(const cuda::std::ptrdiff_t val, const int expected) const noexcept
  {
    assert(val == expected); // asserts that the assigned value matches the index
  }
};

struct mutable_functor
{
  __host__ __device__ constexpr void operator()(const cuda::std::ptrdiff_t val, const int expected) noexcept
  {
    assert(val == expected); // asserts that the assigned value matches the index
  }
};

#endif // TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
