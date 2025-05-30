//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iterator operator-(iterator i, difference_type n);
// friend constexpr difference_type operator-(const iterator& x, const iterator& y);

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr ptrdiff_t operator-(int* lhs, random_access_iterator<int*> rhs)
{
  return lhs - base(rhs);
}

__host__ __device__ constexpr ptrdiff_t operator-(random_access_iterator<int*> lhs, int* rhs)
{
  return base(lhs) - rhs;
}

__host__ __device__ constexpr bool operator==(int* lhs, random_access_iterator<int*> rhs)
{
  return lhs == base(rhs);
}

__host__ __device__ constexpr bool operator==(random_access_iterator<int*> lhs, int* rhs)
{
  return base(lhs) == rhs;
}

__host__ __device__ constexpr bool operator!=(int* lhs, random_access_iterator<int*> rhs)
{
  return lhs != base(rhs);
}

__host__ __device__ constexpr bool operator!=(random_access_iterator<int*> lhs, int* rhs)
{
  return base(lhs) != rhs;
}
static_assert(cuda::std::sized_sentinel_for<int*, random_access_iterator<int*>>);

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};
  { // iter - n
    cuda::strided_iterator iter1{buffer + 7, stride};
    cuda::strided_iterator iter2{buffer + 7, stride};
    assert(iter1 == iter2);
    assert(iter1 - 0 == iter2);
    assert(iter1 - 3 != iter2);
    assert((iter1 - 3).base() == buffer + 7 - 3 * iter1.stride());

    // The original iterator is unchanged
    assert(iter1.base() == buffer + 7);

    static_assert(noexcept(iter2 - 3));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 3)>);
  }

  { // iter - iter
    cuda::strided_iterator iter1{buffer + 1, stride};
    cuda::strided_iterator iter2{buffer + 7, stride};
    assert(iter1 - iter2 == (iter1.base() - iter2.base()) / iter1.stride());
    assert(iter2 - iter1 == (iter2.base() - iter1.base()) / iter1.stride());

    static_assert(noexcept(iter1 - iter2));
    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  { // iter - other_iter
    cuda::strided_iterator iter1{buffer + 1, stride};
    cuda::strided_iterator iter2{random_access_iterator<int*>{buffer + 7}, stride};
    assert(iter1 - iter2 == (iter1.base() - iter2.base()) / iter1.stride());
    assert(iter2 - iter1 == (iter2.base() - iter1.base()) / iter1.stride());

    static_assert(!noexcept(iter1 - iter2));
    static_assert(cuda::std::is_same_v<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
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
