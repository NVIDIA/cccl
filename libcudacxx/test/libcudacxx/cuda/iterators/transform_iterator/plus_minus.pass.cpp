//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_iterator::operator{+,-}

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT can_plus = _CCCL_REQUIRES_EXPR((Iter), Iter i)((i + 42), (42 + i));

template <class Iter>
_CCCL_CONCEPT can_minus = _CCCL_REQUIRES_EXPR((Iter), Iter i)((i - 42));

template <class Iter>
__host__ __device__ constexpr void test()
{
  if constexpr (cuda::std::random_access_iterator<Iter>)
  {
    int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    cuda::transform_iterator iter1{Iter{buffer + 4}, PlusOne{}};
    cuda::transform_iterator iter2{Iter{buffer}, PlusOne{}};

    assert((iter1 + 1).base() == Iter{buffer + 5});
    assert((1 + iter1).base() == Iter{buffer + 5});
    assert((iter1 - 1).base() == Iter{buffer + 3});
    assert(iter1 - iter2 == 4);
    assert((iter1 + 2) - 2 == iter1);
    assert((iter1 - 2) + 2 == iter1);
  }
  else if constexpr (::cuda::std::sized_sentinel_for<Iter, Iter>)
  {
    static_assert(!can_plus<cuda::transform_iterator<PlusOne, Iter>>);
    static_assert(!can_minus<cuda::transform_iterator<PlusOne, Iter>>);

    int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    cuda::transform_iterator iter1{Iter{buffer + 4}, PlusOne{}};
    cuda::transform_iterator iter2{Iter{buffer}, PlusOne{}};
    assert(iter1 - iter2 == 4);
    static_assert(noexcept(iter1 - iter2));
    static_assert(cuda::std::same_as<decltype(iter1 - iter2), cuda::std::iter_difference_t<Iter>>);
  }
  else
  {
    static_assert(!can_plus<cuda::transform_iterator<PlusOne, Iter>>);
    static_assert(!can_minus<cuda::transform_iterator<PlusOne, Iter>>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<forward_sized_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
