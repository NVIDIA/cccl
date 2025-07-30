//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iterator operator+(iterator i, difference_type n);
// friend constexpr iterator operator+(difference_type n, iterator i);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  cuda::strided_iterator iter1{buffer, stride};
  cuda::strided_iterator iter2{buffer, stride};
  assert(iter1 == iter2);
  assert(iter1 + 0 == iter2);
  assert(iter1 + 3 != iter2);
  assert((iter1 + 3).base() == buffer + 3 * iter1.stride());

  // The original iterator is unchanged
  assert(iter1.base() == buffer);

  static_assert(noexcept(iter2 + 3));
  static_assert(!cuda::std::is_reference_v<decltype(iter2 + 3)>);
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
