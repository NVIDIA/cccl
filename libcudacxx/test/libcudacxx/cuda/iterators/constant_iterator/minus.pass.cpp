//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iterator operator-(iterator i, difference_type n)
// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
// friend constexpr iterator operator-(iterator i, default_sentinel)
// friend constexpr iterator operator-(default_sentinel, iterator i)

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test(T value)
{
  { // iterator - difference_type
    cuda::constant_iterator iter1{value, 1337};
    cuda::constant_iterator iter2{value, 1337};
    assert(iter1.index() == 1337);
    assert(iter2.index() == 1337);
    assert(iter1 == iter2);
    assert(iter1 - 0 == iter2);
    assert(iter1 - 5 != iter2);
    assert((iter1 - 5).index() == 1332);

    static_assert(noexcept(iter2 - 5));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
  }

  { // iterator - iterator
    cuda::constant_iterator iter1{value, 10};
    cuda::constant_iterator iter2{value, 5};
    assert(iter1.index() == 10);
    assert(iter2.index() == 5);
    assert(iter1 != iter2);
    assert(iter1 - iter2 == 5);
    assert(iter1 - iter1 == 0);
    assert(iter2 - iter1 == -5);

    static_assert(noexcept(iter2 - 5));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
  }

  { // iterator - default_sentinel
    cuda::constant_iterator iter1{value, 10};
    assert(iter1.index() == 10);
    assert(iter1 - cuda::std::default_sentinel == -10);
    assert(cuda::std::default_sentinel - iter1 == 10);

    static_assert(noexcept(iter1 - cuda::std::default_sentinel));
    static_assert(noexcept(cuda::std::default_sentinel - iter1));
    static_assert(!cuda::std::is_reference_v<decltype(iter1 - cuda::std::default_sentinel)>);
    static_assert(!cuda::std::is_reference_v<decltype(cuda::std::default_sentinel - iter1)>);
  }
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
