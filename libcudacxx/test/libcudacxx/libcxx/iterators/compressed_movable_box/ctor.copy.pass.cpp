//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Arithmetic operators

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

template <class... Ts>
using box = cuda::std::__compressed_movable_box<Ts...>;

template <class T>
__host__ __device__ constexpr void test(const int expected)
{
  constexpr bool is_nothrow = cuda::std::is_nothrow_copy_constructible_v<T>;
  { // single item
    const box<T> input{1337};
    box<T> b{input};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T>> == is_nothrow);
  }
  { // two items
    const box<T, int> input{1337};
    box<T, int> b{input};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T, int>> == is_nothrow);
  }
  { // three items
    const box<T, int, int> input{1337};
    box<T, int, int> b{input};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T, int, int>> == is_nothrow);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  { // trivial empty type
    using T = TrivialEmpty;
    const box<T> input{};
    box<T> b{input};
    assert(b.__get<0>() == 42);
    static_assert(cuda::std::is_nothrow_copy_constructible_v<box<T>>);
  }

  { // trivial nonempty type
    test<int>(1337);
  }

  { // non-trivial empty type
    test<NotTriviallyCopyConstructibleEmpty<42>>(42);
  }

  { // non-trivial nonempty type
    test<NotTriviallyCopyConstructible<42>>(1337);
  }

  { // non-trivial empty type, not noexcept
    test<NotTriviallyCopyConstructibleEmpty<MayThrow>>(MayThrow);
  }

  { // non-trivial nonempty type, not noexcept
    test<NotTriviallyCopyConstructible<MayThrow>>(1337);
  }

  { // not default constructible
    test<NotDefaultConstructible>(1337);
  }

  { // not copy assignable not default constructible
    test<NotCopyAssignableNotDefaultConstructible<42>>(1337);
  }

  { // not copy constructible
    static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructible<42>>>);
    static_assert(!cuda::std::is_copy_constructible_v<box<NotCopyConstructibleEmpty>>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
