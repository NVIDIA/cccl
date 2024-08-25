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

// transform_view::<iterator>::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/std/ranges>
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  {
    // Test a new-school iterator with operator<=>; the transform iterator should also have operator<=>.
    using It = three_way_contiguous_iterator<int*>;
    static_assert(cuda::std::three_way_comparable<It>);
    using R = cuda::std::ranges::transform_view<cuda::std::ranges::subrange<It>, PlusOne>;
    static_assert(cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);

    int a[] = {1, 2, 3};
    cuda::std::same_as<R> auto r =
      cuda::std::ranges::subrange<It>(It(a), It(a + 3)) | cuda::std::views::transform(PlusOne());
    auto iter1 = r.begin();
    auto iter2 = iter1 + 1;

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

    assert((iter1 <=> iter2) == cuda::std::strong_ordering::less);
    assert((iter1 <=> iter1) == cuda::std::strong_ordering::equal);
    assert((iter2 <=> iter1) == cuda::std::strong_ordering::greater);
  }
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have operator<=> either.
    using It = random_access_iterator<int*>;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    static_assert(!cuda::std::three_way_comparable<It>);
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR
    using R = cuda::std::ranges::transform_view<cuda::std::ranges::subrange<It>, PlusOne>;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

    int a[] = {1, 2, 3};
    cuda::std::ranges::subrange<It> input(It(a), It(a + 3));
    decltype(auto) result = input | cuda::std::views::transform(PlusOne());
    static_assert(cuda::std::same_as<decltype(result), R>);
    auto iter1 = result.begin();
    auto iter2 = iter1 + 1;

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
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
