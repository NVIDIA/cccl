//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// counting_iterator::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/iterator>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include "test_macros.h"
#include "types.h"

template <typename T>
TEST_FUNC constexpr void test()
{
  cuda::counting_iterator<T> iter1{T{42}};
  const auto iter2 = iter1 + 1;

  assert(!(iter1 < iter1));
  assert(iter1 < iter2);
  assert(!(iter2 < iter1));
  assert(iter1 <= iter1);
  assert(iter1 <= iter2);
  assert(!(iter2 <= iter1));
  assert(!(iter1 > iter1));
  assert(!(iter1 > iter2));
  assert(iter2 > iter1);
  assert(iter1 >= iter1);
  assert(!(iter1 >= iter2));
  assert(iter2 >= iter1);
  assert(iter1 == iter1);
  assert(!(iter1 == iter2));
  assert(iter2 == iter2);
  assert(!(iter1 != iter1));
  assert(iter1 != iter2);
  assert(!(iter2 != iter2));

#if TEST_HAS_SPACESHIP()
  static_assert(cuda::std::three_way_comparable<cuda::counting_iterator<T>>);
  assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
  assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
  assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
#endif // TEST_HAS_SPACESHIP()
}

TEST_FUNC constexpr bool test()
{
  test<SomeInt>();
  test<cuda::std::int8_t>();
  test<cuda::std::uint8_t>();
  test<int>();
  test<cuda::std::int64_t>();
  test<cuda::std::uint64_t>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
