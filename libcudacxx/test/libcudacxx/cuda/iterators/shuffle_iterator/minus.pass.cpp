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
//   requires advanceable<W>;
// friend constexpr difference_type operator-(const iterator& x, const iterator& y)
//   requires advanceable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  // <iterator> - difference_type
  {
    cuda::shuffle_iterator iter1{fake_bijection{}, 3};
    cuda::shuffle_iterator iter2{fake_bijection{}, 3};
    assert(iter1 == iter2);
    assert(iter1 - 0 == iter2);
    assert(iter1 - 2 != iter2);
    assert(iter1 - 2 == cuda::std::ranges::prev(iter2, 2));

    static_assert(noexcept(iter2 - 2));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 2)>);
  }

  // <iterator> - <iterator>
  {
    cuda::shuffle_iterator iter1{fake_bijection{}, 5};
    cuda::shuffle_iterator iter2{fake_bijection{}, 0};
    assert(iter1 - iter2 == 5);
    assert(iter1 - iter1 == 0);
    assert(iter2 - iter1 == -5);

    using shuffle_iter = decltype(iter1);
    static_assert(noexcept(iter1 - iter2));
    static_assert(cuda::std::same_as<decltype(iter1 - iter2), typename shuffle_iter::difference_type>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
