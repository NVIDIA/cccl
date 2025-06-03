//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr explicit strided_iterator(iter, stride);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  { // CTAD
    cuda::strided_iterator iter{buffer, stride};
    assert(iter.stride() == stride);
    assert(iter.base() == buffer);
  }

  {
    cuda::strided_iterator<int*, Stride> iter{buffer, stride};
    assert(iter.stride() == stride);
    assert(iter.base() == buffer);
  }

  // Cannot construct a strided_iterator with an integer stride from just an iterator
  if constexpr (!cuda::std::__integer_like<Stride>)
  {
    cuda::strided_iterator<int*, Stride> iter{buffer};
    assert(iter.stride() == stride);
  }
}

__host__ __device__ constexpr bool test()
{
  test(2);
  test(Stride<2>{});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
