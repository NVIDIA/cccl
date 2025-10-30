//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator--() requires all-bidirectional<Const, Views...>;
// constexpr iterator operator--(int) requires all-bidirectional<Const, Views...>;

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT canDecrement = _CCCL_REQUIRES_EXPR((Iter), Iter iter)((--iter), (iter--));

__host__ __device__ constexpr bool test()
{
  int a[]    = {1, 2, 3, 4};
  double b[] = {4.1, 3.2, 4.3};

  { // all random_access_iterator
    cuda::zip_transform_iterator iter{Plus{}, a + 3, random_access_iterator{b + 3}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(--iter), Iter&>);
    auto& it_ref = --iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));

    const int expected1 = a[2] + static_cast<int>(b[2]);
    assert(*iter == expected1);

    static_assert(cuda::std::is_same_v<decltype(iter--), Iter>);
    iter--;
    const int expected2 = a[1] + static_cast<int>(b[1]);
    assert(*iter == expected2);
  }

  { // all bidirectional_iterator
    cuda::zip_transform_iterator iter{Plus{}, a + 3, bidirectional_iterator{b + 3}};
    using Iter = decltype(iter);

    static_assert(cuda::std::is_same_v<decltype(--iter), Iter&>);
    auto& it_ref = --iter;
    assert(cuda::std::addressof(it_ref) == cuda::std::addressof(iter));
    const int expected1 = a[2] + static_cast<int>(b[2]);
    assert(*iter == expected1);

    static_assert(cuda::std::is_same_v<decltype(iter--), Iter>);
    iter--;
    const int expected2 = a[1] + static_cast<int>(b[1]);
    assert(*iter == expected2);
  }

  { // not all bidirectional_iterator
    cuda::zip_transform_iterator iter{Plus{}, a + 3, cpp17_input_iterator{b + 3}};
    using Iter = decltype(iter);

    static_assert(!canDecrement<Iter>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
