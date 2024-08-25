//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// repeat_view::<iterator>::operator{==,<=>}

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // Test unbound
  {
    using R = cuda::std::ranges::repeat_view<int>;
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

    R r(42);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    static_assert(cuda::std::same_as<decltype(iter1 == iter2), bool>);

    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);

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

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);

    static_assert(cuda::std::same_as<decltype(iter1 <=> iter2), cuda::std::strong_ordering>);
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  }

  // Test bound
  {
    using R = cuda::std::ranges::repeat_view<int, int>;
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

    R r(42, 10);
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

    static_assert(cuda::std::same_as<decltype(iter1 == iter2), bool>);

    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
    assert(iter2 == iter2);

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

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  }

  return true;
}

int main(int, char**)
{
  test();
#if !defined(TEST_COMPILER_CLANG) || __clang__ > 9
  static_assert(test());
#endif // clang > 9

  return 0;
}
