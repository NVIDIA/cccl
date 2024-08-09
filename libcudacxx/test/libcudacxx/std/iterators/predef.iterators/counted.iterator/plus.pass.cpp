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

// constexpr counted_iterator operator+(iter_difference_t<I> n) const
//     requires random_access_iterator<I>;
// friend constexpr counted_iterator operator+(
//   iter_difference_t<I> n, const counted_iterator& x)
//     requires random_access_iterator<I>;
// constexpr counted_iterator& operator+=(iter_difference_t<I> n)
//     requires random_access_iterator<I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class Iter>
concept PlusEnabled = requires(Iter& iter) { iter + 1; };

template <class Iter>
concept PlusEqEnabled = requires(Iter& iter) { iter += 1; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class, class = void>
inline constexpr bool PlusEnabled = false;

template <class Iter>
inline constexpr bool PlusEnabled<Iter, cuda::std::void_t<decltype(cuda::std::declval<Iter&>() += 1)>> = true;

template <class, class = void>
inline constexpr bool PlusEqEnabled = false;

template <class Iter>
inline constexpr bool PlusEqEnabled<Iter, cuda::std::void_t<decltype(cuda::std::declval<Iter&>() += 1)>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    {
      using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
      cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(iter + 2 == Counted(random_access_iterator<int*>{buffer + 2}, 6));
      assert(iter + 0 == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), Counted);
    }
    {
      using Counted = const cuda::std::counted_iterator<random_access_iterator<int*>>;
      const cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(iter + 8 == Counted(random_access_iterator<int*>{buffer + 8}, 0));
      assert(iter + 0 == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), cuda::std::remove_const_t<Counted>);
    }
  }

  {
    {
      using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
      cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(2 + iter == Counted(random_access_iterator<int*>{buffer + 2}, 6));
      assert(0 + iter == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), Counted);
    }
    {
      using Counted = const cuda::std::counted_iterator<random_access_iterator<int*>>;
      const cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert(8 + iter == Counted(random_access_iterator<int*>{buffer + 8}, 0));
      assert(0 + iter == Counted(random_access_iterator<int*>{buffer}, 8));

      ASSERT_SAME_TYPE(decltype(iter + 2), cuda::std::remove_const_t<Counted>);
    }
  }

  {
    {
      using Counted = cuda::std::counted_iterator<random_access_iterator<int*>>;
      cuda::std::counted_iterator iter(random_access_iterator<int*>{buffer}, 8);
      assert((iter += 2) == Counted(random_access_iterator<int*>{buffer + 2}, 6));
      assert((iter += 0) == Counted(random_access_iterator<int*>{buffer + 2}, 6));

      ASSERT_SAME_TYPE(decltype(iter += 2), Counted&);
    }
    {
      using Counted = cuda::std::counted_iterator<contiguous_iterator<int*>>;
      cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
      assert((iter += 8) == Counted(contiguous_iterator<int*>{buffer + 8}, 0));
      assert((iter += 0) == Counted(contiguous_iterator<int*>{buffer + 8}, 0));

      ASSERT_SAME_TYPE(decltype(iter += 2), Counted&);
    }
    {
      static_assert(PlusEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
      static_assert(!PlusEnabled<cuda::std::counted_iterator<bidirectional_iterator<int*>>>);

      static_assert(PlusEqEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
      static_assert(!PlusEqEnabled<const cuda::std::counted_iterator<random_access_iterator<int*>>>);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
