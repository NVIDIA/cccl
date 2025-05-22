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

// If we're compiling for 32 bit or windows, int and long are the same size, so long long is the correct difference
// type.
#if INTPTR_MAX == INT32_MAX || defined(_WIN32)
using IntDiffT = long long;
#else
using IntDiffT = long;
#endif

__host__ __device__ constexpr bool test()
{
  // <iterator> - difference_type
  {
    { // When "_Start" is signed integer like.
      cuda::counting_iterator<int> iter1{10};
      cuda::counting_iterator<int> iter2{10};
      assert(iter1 == iter2);
      assert(iter1 - 0 == iter2);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == cuda::std::ranges::prev(iter2, 5));

      static_assert(noexcept(iter2 - 5));
      static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
    }

    { // When "_Start" is not integer like.
      cuda::counting_iterator<SomeInt> iter1{SomeInt{10}};
      cuda::counting_iterator<SomeInt> iter2{SomeInt{10}};
      assert(iter1 == iter2);
      assert(iter1 - 0 == iter2);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == cuda::std::ranges::prev(iter2, 5));

      static_assert(!noexcept(iter2 - 5));
      static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
    }

    { // When "_Start" is unsigned integer like and n is greater than or equal to zero.
      cuda::counting_iterator<unsigned> iter1{10};
      cuda::counting_iterator<unsigned> iter2{10};
      assert(iter1 == iter2);
      assert(iter1 - 0 == iter2);
      assert(iter1 - 5 != iter2);
      assert(iter1 - 5 == cuda::std::ranges::prev(iter2, 5));

      static_assert(noexcept(iter2 - 5));
      static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
    }

    { // When "_Start" is unsigned integer like and n is less than zero.
      cuda::counting_iterator<unsigned> iter1{10};
      cuda::counting_iterator<unsigned> iter2{10};
      assert(iter1 == iter2);
      assert(iter1 - (-5) != iter2);
      assert(iter1 - (-5) == cuda::std::ranges::next(iter2, 5));

      static_assert(noexcept(iter2 - (-5)));
      static_assert(!cuda::std::is_reference_v<decltype(iter2 - (-5))>);
    }
  }

  // <iterator> - <iterator>
  {
    { // When "_Start" is signed integer like.
      cuda::counting_iterator<int> iter1{10};
      cuda::counting_iterator<int> iter2{5};
      assert(iter1 - iter2 == 5);
      assert(iter1 - iter1 == 0);
      assert(iter2 - iter1 == -5);

      static_assert(noexcept(iter1 - iter2));
      static_assert(cuda::std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }

    { // When "_Start" is signed integer like.
      cuda::counting_iterator<unsigned> iter1{10};
      cuda::counting_iterator<unsigned> iter2{5};
      assert(iter1 - iter2 == 5);
      assert(iter1 - iter1 == 0);
      assert(iter2 - iter1 == -5);

      static_assert(noexcept(iter1 - iter2));
      static_assert(cuda::std::same_as<decltype(iter1 - iter2), IntDiffT>);
    }

    { // When "_Start" is not integer like.
      cuda::counting_iterator<SomeInt> iter1{SomeInt{10}};
      cuda::counting_iterator<SomeInt> iter2{SomeInt{5}};
      assert(iter1 - iter2 == 5);
      assert(iter1 - iter1 == 0);
      assert(iter2 - iter1 == -5);

      static_assert(!noexcept(iter1 - iter2));
      static_assert(cuda::std::same_as<decltype(iter1 - iter2), int>);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
