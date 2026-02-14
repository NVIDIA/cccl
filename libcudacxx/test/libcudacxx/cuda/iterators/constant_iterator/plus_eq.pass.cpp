//===---------------------------------------------------------------------+===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------+===//

// constexpr iterator& operator+=(difference_type n)

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test(T value)
{
  cuda::constant_iterator iter1{value, 1337};
  cuda::constant_iterator iter2{value, 1337};
  assert(iter1.index() == 1337);
  assert(iter2.index() == 1337);
  assert(iter1 == iter2);
  iter1 += 0;
  assert(iter1 == iter2);
  assert(iter1.index() == 1337);
  iter1 += 5;
  assert(iter1 != iter2);
  assert(iter1.index() == 1342);

  static_assert(noexcept(iter2 += 5));
  static_assert(cuda::std::is_reference_v<decltype(iter2 += 5)>);
}

__host__ __device__ constexpr bool test()
{
  test(42);
  test(NotDefaultConstructible{42});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
