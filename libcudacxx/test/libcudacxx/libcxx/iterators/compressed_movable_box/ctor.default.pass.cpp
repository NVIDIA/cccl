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
  constexpr bool is_nothrow = cuda::std::is_nothrow_default_constructible_v<T>;
  { // single item
    const box<T> b{};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_default_constructible_v<box<T>> == is_nothrow);
  }
  { // two items
    const box<T, int> b{};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_default_constructible_v<box<T, int>> == is_nothrow);
  }
  { // three items
    const box<T, int, int> b{};
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_default_constructible_v<box<T, int, int>> == is_nothrow);
  }
}

__host__ __device__ constexpr bool test()
{
  { // trivial nonempty type
    test<int>(0);
  }

  { // non-trivial empty type
    test<NotTriviallyDefaultConstructibleEmpty<42>>(42);
  }

  { // non-trivial nonempty type
    test<NotTriviallyDefaultConstructible<42>>(42);
  }

  { // non-trivial empty type, not noexcept
    test<NotTriviallyDefaultConstructibleEmpty<MayThrow>>(MayThrow);
  }

  { // non-trivial nonempty type, not noexcept
    test<NotTriviallyDefaultConstructible<MayThrow>>(MayThrow);
  }

  { // Not copyable
    test<NotCopyConstructible<42>>(42);
  }

  { // Not copy-assignable
    test<NotCopyAssignable<42>>(42);
  }

  { // Not move-assignable
    test<NotMoveAssignable<42>>(42);
  }

  { // not default constructible
    static_assert(!cuda::std::is_default_constructible_v<NotDefaultConstructible>);
    static_assert(!cuda::std::is_default_constructible_v<box<NotDefaultConstructible>>);
    static_assert(!cuda::std::is_default_constructible_v<box<NotDefaultConstructible, int>>);
    static_assert(!cuda::std::is_default_constructible_v<box<NotDefaultConstructible, int, int>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
