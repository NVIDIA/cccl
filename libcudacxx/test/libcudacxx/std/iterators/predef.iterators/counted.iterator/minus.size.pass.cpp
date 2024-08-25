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

// constexpr counted_iterator operator-(iter_difference_t<I> n) const
//   requires random_access_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class Iter>
concept MinusEnabled = requires(Iter& iter) { iter - 1; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class, class = void>
inline constexpr bool MinusEnabled = false;

template <class Iter>
inline constexpr bool MinusEnabled<Iter, cuda::std::void_t<decltype(cuda::std::declval<Iter&>() - 1)>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
    Counted iter(random_access_iterator<int*>{buffer + 2}, 6);
    assert(iter - 2 == Counted(random_access_iterator<int*>{buffer}, 8));
    assert(iter - 0 == Counted(random_access_iterator<int*>{buffer + 2}, 6));
    assert(iter.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter - 2), Counted);
  }
  {
    using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
    const Counted iter(random_access_iterator<int*>{buffer + 2}, 6);
    assert(iter - 2 == Counted(random_access_iterator<int*>{buffer}, 8));
    assert(iter - 0 == Counted(random_access_iterator<int*>{buffer + 2}, 6));
    assert(iter.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter - 2), Counted);
  }
  {
    using Counted = cuda::std::counted_iterator<contiguous_iterator<int*>>;
    Counted iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter - 2 == Counted(contiguous_iterator<int*>{buffer}, 8));
    assert(iter - 0 == Counted(contiguous_iterator<int*>{buffer + 2}, 6));
    assert(iter.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter - 2), Counted);
  }
  {
    using Counted = cuda::std::counted_iterator<contiguous_iterator<int*>>;
    const Counted iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter - 2 == Counted(contiguous_iterator<int*>{buffer}, 8));
    assert(iter - 0 == Counted(contiguous_iterator<int*>{buffer + 2}, 6));
    assert(iter.count() == 6);

    ASSERT_SAME_TYPE(decltype(iter - 2), Counted);
  }

  {
    static_assert(MinusEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
    static_assert(!MinusEnabled<cuda::std::counted_iterator<bidirectional_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
