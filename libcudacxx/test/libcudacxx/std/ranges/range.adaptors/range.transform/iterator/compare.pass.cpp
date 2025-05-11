//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_view::<iterator>::operator{<,>,<=,>=,==,!=,<=>}

#include <cuda/std/ranges>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
#if TEST_HAS_SPACESHIP()
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
#endif // TEST_HAS_SPACESHIP()

  {
    // Test an old-school iterator with no operator<=>; the transform iterator shouldn't have operator<=> either.
    using It = random_access_iterator<int*>;
#if TEST_HAS_SPACESHIP()
    static_assert(!cuda::std::three_way_comparable<It>);
#endif // TEST_HAS_SPACESHIP()
    using R = cuda::std::ranges::transform_view<cuda::std::ranges::subrange<It>, PlusOne>;
#if TEST_HAS_SPACESHIP()
    static_assert(!cuda::std::three_way_comparable<cuda::std::ranges::iterator_t<R>>);
#endif // TEST_HAS_SPACESHIP()

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
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
