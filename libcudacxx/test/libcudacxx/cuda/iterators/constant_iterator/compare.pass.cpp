//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constant_iterator::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/iterator>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::constant_iterator iter1{42, 2};
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

    static_assert(noexcept(iter1 == iter2));
    static_assert(noexcept(iter1 != iter2));
    static_assert(noexcept(iter1 < iter2));
    static_assert(noexcept(iter1 <= iter2));
    static_assert(noexcept(iter1 > iter2));
    static_assert(noexcept(iter1 >= iter2));

#if TEST_HAS_SPACESHIP()
    static_assert(cuda::std::three_way_comparable<cuda::constant_iterator<int>>);
    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
#endif // TEST_HAS_SPACESHIP()
  }

  { // against cuda::std::default_sentinel
    cuda::constant_iterator iter1{42, 0};
    cuda::constant_iterator iter2{42, 1};
    assert(iter1 == cuda::std::default_sentinel);
    assert(!(iter1 != cuda::std::default_sentinel));
    assert(!(iter2 == cuda::std::default_sentinel));
    assert(iter2 != cuda::std::default_sentinel);

    assert(cuda::std::default_sentinel == iter1);
    assert(!(cuda::std::default_sentinel != iter1));
    assert(!(cuda::std::default_sentinel == iter2));
    assert(cuda::std::default_sentinel != iter2);

    static_assert(noexcept(iter1 == cuda::std::default_sentinel));
    static_assert(noexcept(iter1 != cuda::std::default_sentinel));
    static_assert(noexcept(cuda::std::default_sentinel == iter1));
    static_assert(noexcept(cuda::std::default_sentinel != iter1));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
