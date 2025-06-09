//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// index() const noexcept

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test()
{
  {
    const T val = 42;
    const cuda::constant_iterator iter{val};
    assert(iter.index() == 0);
    static_assert(noexcept(iter.index()));
    static_assert(cuda::std::is_same_v<decltype(iter.index()), typename cuda::constant_iterator<T>::difference_type>);
  }

  {
    const T val = 42;
    const cuda::constant_iterator iter{val, 1337};
    assert(*iter == T{42});
    assert(iter.index() == 1337);
    static_assert(noexcept(iter.index()));
    static_assert(cuda::std::is_same_v<decltype(iter.index()), typename cuda::constant_iterator<T>::difference_type>);
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
