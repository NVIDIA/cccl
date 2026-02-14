//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr explicit iterator(W value);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test()
{
  { // CTAD
    const T val = 42;
    cuda::constant_iterator iter{val};
    assert(*iter == T{42});
    assert(iter.index() == 0);

    static_assert(cuda::std::is_same_v<decltype(iter), cuda::constant_iterator<T, cuda::std::ptrdiff_t>>);
  }

  { // CTAD
    const T val     = 42;
    const int index = 1337;
    cuda::constant_iterator iter{val, index};
    assert(*iter == T{42});
    assert(iter.index() == index);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::constant_iterator<T, int>>);
  }

  { // CTAD with explicit integer type
    const T val       = 42;
    const short index = 1337;
    cuda::constant_iterator iter{val, index};
    assert(*iter == T{42});
    assert(iter.index() == index);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::constant_iterator<T, short>>);
  }

  {
    const T val = 42;
    cuda::constant_iterator<T> iter{val};
    assert(*iter == T{42});
    assert(iter.index() == 0);
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::constant_iterator<T, cuda::std::ptrdiff_t>>);
  }

  {
    const T val       = 42;
    const short index = 1337;
    cuda::constant_iterator<T, int> iter{val, index};
    assert(*iter == T{42});
    assert(iter.index() == index);
  }

  {
    const T val     = 42;
    const int index = 1337;
    cuda::constant_iterator<T, int> iter{val, index};
    assert(*iter == T{42});
    assert(iter.index() == index);
  }

  { // explicit index type with different argument type
    const T val     = 42;
    const int index = 1337;
    cuda::constant_iterator<T, short> iter{val, index};
    assert(*iter == T{42});
    assert(iter.index() == index);
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
