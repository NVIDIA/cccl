//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <cuda/std/span>

// Conversion from a type that is *not* a range. We had a bug where we would still try to instantiate `iterator_t`,
// which would fail because of a missing `begin`

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

struct ConvertibleButNoRange
{
  int buffer[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  __host__ __device__ constexpr operator cuda::std::span<int>() const noexcept
  {
    return cuda::std::span<int>{const_cast<int*>(buffer), 10};
  }
};

__host__ __device__ constexpr bool test()
{
  ConvertibleButNoRange input{};
  cuda::std::span<int> converted = input;
  assert(converted.data() == input.buffer);
  assert(converted.size() == 10);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
