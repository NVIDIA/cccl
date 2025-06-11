//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator() requires default_initializable<T> = default;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test()
{
  if constexpr (cuda::std::is_default_constructible_v<T>)
  {
    cuda::constant_iterator<T> iter;
    assert(*iter == T{});
    assert(iter.index() == 0);
  }
  else
  {
    static_assert(!cuda::std::is_default_constructible_v<cuda::constant_iterator<T>>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<NotDefaultConstructible>();
  test<DefaultConstructibleTo42>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
