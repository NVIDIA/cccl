//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr make_strided_iterator(iter, stride = 0);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  { // make_strided_iterator(iter, stride);
    cuda::strided_iterator iter = cuda::make_strided_iterator(buffer, stride);
    assert(iter.stride() == stride);
    assert(iter.base() == buffer);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::strided_iterator<int*, Stride>>);
  }

  { // explicit template arguments make_strided_iterator(iter, stride);
    cuda::strided_iterator iter = cuda::make_strided_iterator<const int*, int>(buffer, static_cast<int>(stride));
    assert(iter.stride() == stride);
    assert(iter.base() == buffer);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::strided_iterator<const int*, int>>);
  }

  if constexpr (cuda::std::__integral_constant_like<Stride>)
  {
    { // make_strided_iterator(iter);
      cuda::strided_iterator iter = cuda::make_strided_iterator<Stride>(buffer);
      assert(iter.stride() == 2);
      assert(iter.base() == buffer);
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::strided_iterator<int*, Stride>>);
    }

    { // explicit template arguments make_strided_iterator(iter);
      cuda::strided_iterator iter = cuda::make_strided_iterator<Stride, const int*>(buffer);
      assert(iter.stride() == 2);
      assert(iter.base() == buffer);
      static_assert(cuda::std::is_same_v<decltype(iter), cuda::strided_iterator<const int*, Stride>>);
    }
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
  static_assert(test());

  return 0;
}
