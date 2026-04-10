//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator& operator-=(difference_type n)
//   requires advanceable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  { // When "_Start" is signed integer like.
    cuda::counting_iterator<int> iter1{10};
    cuda::counting_iterator<int> iter2{10};
    assert(iter1 == iter2);
    iter1 -= 0;
    assert(iter1 == iter2);
    iter1 -= 5;
    assert(iter1 != iter2);
    assert(iter1 == cuda::std::ranges::prev(iter2, 5));

    static_assert(noexcept(iter2 -= 5));
    static_assert(cuda::std::is_reference_v<decltype(iter2 -= 5)>);
  }

  { // When "_Start" is not integer like.
    cuda::counting_iterator<SomeInt> iter1{SomeInt{10}};
    cuda::counting_iterator<SomeInt> iter2{SomeInt{10}};
    assert(iter1 == iter2);
    iter1 -= 0;
    assert(iter1 == iter2);
    iter1 -= 5;
    assert(iter1 != iter2);
    assert(iter1 == cuda::std::ranges::prev(iter2, 5));

    static_assert(!noexcept(iter2 -= 5));
    static_assert(cuda::std::is_reference_v<decltype(iter2 -= 5)>);
  }

  { // When "_Start" is unsigned integer like and n is greater than or equal to zero.
    cuda::counting_iterator<unsigned> iter1{10};
    cuda::counting_iterator<unsigned> iter2{10};
    assert(iter1 == iter2);
    iter1 -= 0;
    assert(iter1 == iter2);
    iter1 -= 5;
    assert(iter1 != iter2);
    assert(iter1 == cuda::std::ranges::prev(iter2, 5));

    static_assert(noexcept(iter2 -= 5));
    static_assert(cuda::std::is_reference_v<decltype(iter2 -= 5)>);
  }

  { // When "_Start" is unsigned integer like and n is less than zero.
    using difference_type = typename cuda::counting_iterator<unsigned>::difference_type;
    cuda::counting_iterator<unsigned> iter1{10};
    cuda::counting_iterator<unsigned> iter2{10};
    assert(iter1 == iter2);
    iter1 -= difference_type(-5);
    assert(iter1 != iter2);
    assert(iter1 == cuda::std::ranges::next(iter2, 5));

    static_assert(noexcept(iter2 -= -5));
    static_assert(cuda::std::is_reference_v<decltype(iter2 -= difference_type(-5))>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
