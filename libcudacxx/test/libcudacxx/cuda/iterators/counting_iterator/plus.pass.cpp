//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iterator operator+(iterator i, difference_type n)
//   requires advanceable<W>;
// friend constexpr iterator operator+(difference_type n, iterator i)
//   requires advanceable<W>;

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

TEST_FUNC constexpr bool test()
{
  { // When "_Start" is signed integer like.
    cuda::counting_iterator<int> iter1{0};
    cuda::counting_iterator<int> iter2{0};
    assert(iter1 == iter2);
    assert(iter1 + 0 == iter1);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(noexcept(iter2 + 5));
    static_assert(cuda::std::is_same_v<decltype(iter2 + (-5)), decltype(iter2)>);
  }

  { // When "_Start" is signed integer like, explicit difference type.
    cuda::counting_iterator<int, cuda::std::int8_t> iter1{0};
    cuda::counting_iterator<int, cuda::std::int8_t> iter2{0};
    assert(iter1 == iter2);
    assert(iter1 + cuda::std::int8_t{0} == iter1);
    assert(iter1 + cuda::std::int8_t{5} != iter2);
    assert(iter1 + cuda::std::int8_t{5} == cuda::std::ranges::next(iter2, 5));

    static_assert(noexcept(iter2 + cuda::std::int8_t{5}));
    static_assert(cuda::std::is_same_v<decltype(iter2 + cuda::std::int8_t{-5}), decltype(iter2)>);
  }

  { // When "_Start" is not integer like.
    static_assert(cuda::std::totally_ordered<SomeInt>);
    cuda::counting_iterator<SomeInt> iter1{SomeInt{0}};
    cuda::counting_iterator<SomeInt> iter2{SomeInt{0}};
    assert(iter1 == iter2);
    assert(iter1 + 0 == iter1);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(!noexcept(iter2 + 5));
    static_assert(cuda::std::is_same_v<decltype(iter2 + (-5)), decltype(iter2)>);
  }

  { // When "_Start" is not integer like, explicit difference type.
    cuda::counting_iterator<SomeInt, cuda::std::int16_t> iter1{SomeInt{0}};
    cuda::counting_iterator<SomeInt, cuda::std::int16_t> iter2{SomeInt{0}};
    assert(iter1 == iter2);
    assert(iter1 + cuda::std::int16_t{0} == iter1);
    assert(iter1 + cuda::std::int16_t{5} != iter2);
    assert(iter1 + cuda::std::int16_t{5} == cuda::std::ranges::next(iter2, 5));

    static_assert(!noexcept(iter2 + cuda::std::int16_t{5}));
    static_assert(cuda::std::is_same_v<decltype(iter2 + cuda::std::int16_t{-5}), decltype(iter2)>);
  }

  { // When "_Start" is unsigned integer like and n is greater than or equal to zero.
    cuda::counting_iterator<unsigned> iter1{0};
    cuda::counting_iterator<unsigned> iter2{0};
    assert(iter1 == iter2);
    assert(iter1 + 0 == iter1);
    assert(iter1 + 5 != iter2);
    assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));

    static_assert(noexcept(iter2 + 5));
    static_assert(cuda::std::is_same_v<decltype(iter2 + (-5)), decltype(iter2)>);
  }

  { // When "_Start" is unsigned integer like and n is greater than or equal to zero, explicit difference type.
    cuda::counting_iterator<unsigned, cuda::std::int8_t> iter1{0};
    cuda::counting_iterator<unsigned, cuda::std::int8_t> iter2{0};
    assert(iter1 == iter2);
    assert(iter1 + cuda::std::int8_t{0} == iter1);
    assert(iter1 + cuda::std::int8_t{5} != iter2);
    assert(iter1 + cuda::std::int8_t{5} == cuda::std::ranges::next(iter2, 5));

    static_assert(noexcept(iter2 + cuda::std::int8_t{5}));
    static_assert(cuda::std::is_same_v<decltype(iter2 + cuda::std::int8_t{-5}), decltype(iter2)>);
  }

  { // When "_Start" is unsigned integer like and n is less than zero.
    using difference_type = typename cuda::counting_iterator<unsigned>::difference_type;
    cuda::counting_iterator<unsigned> iter1{10};
    cuda::counting_iterator<unsigned> iter2{10};
    assert(iter1 == iter2);
    assert(iter1 + difference_type(-0) == iter1);
    assert(iter1 + difference_type(-5) != iter2);
    assert(iter1 + difference_type(-5) == cuda::std::ranges::prev(iter2, 5));

    static_assert(noexcept(iter2 + (-5)));
    static_assert(cuda::std::is_same_v<decltype(iter2 + (-5)), decltype(iter2)>);
  }

  { // When "_Start" is unsigned integer like and n is less than zero, explicit difference type.
    using difference_type = typename cuda::counting_iterator<unsigned, cuda::std::int8_t>::difference_type;
    cuda::counting_iterator<unsigned, cuda::std::int8_t> iter1{10};
    cuda::counting_iterator<unsigned, cuda::std::int8_t> iter2{10};
    assert(iter1 == iter2);
    assert(iter1 + difference_type(-0) == iter1);
    assert(iter1 + difference_type(-5) != iter2);
    assert(iter1 + difference_type(-5) == cuda::std::ranges::prev(iter2, 5));

    static_assert(noexcept(iter2 + difference_type(-5)));
    static_assert(cuda::std::is_same_v<decltype(iter2 + difference_type(-5)), decltype(iter2)>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
