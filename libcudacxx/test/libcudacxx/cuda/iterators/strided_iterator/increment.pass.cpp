//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator++();
// constexpr iterator operator++(int);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Stride>
__host__ __device__ constexpr void test(Stride stride)
{
  int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8};

  cuda::strided_iterator iter1{buffer + 4, stride};
  cuda::strided_iterator iter2{buffer + 4, stride};
  assert(iter1.stride() == stride);
  assert(iter1 == iter2);
  assert(++iter1 != iter2++);
  assert(iter1 == iter2);
  assert(iter1.base() == buffer + 4 + iter1.stride());
  assert(iter1.stride() == stride);

  static_assert(noexcept(++iter2));
  static_assert(noexcept(iter2++));
  static_assert(!cuda::std::is_reference_v<decltype(iter2++)>);
  static_assert(cuda::std::is_reference_v<decltype(++iter2)>);
  static_assert(cuda::std::same_as<cuda::std::remove_reference_t<decltype(++iter2)>, decltype(iter2++)>);
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
