//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_iterator::operator[]

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT can_subscript = _CCCL_REQUIRES_EXPR((Iter), Iter i)((i[0]));

template <class Iter>
__host__ __device__ constexpr void test()
{
  if constexpr (cuda::std::random_access_iterator<Iter>)
  {
    int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    {
      cuda::transform_iterator iter{Iter{buffer}, PlusOne{}};
      assert(iter[4] == 5);
      assert(cuda::std::as_const(iter)[4] == 5);
      assert(buffer[4] == 4);

      static_assert(!noexcept(iter[4]));
      static_assert(cuda::std::is_same_v<int, decltype(iter[4])>);
    }

    {
      cuda::transform_iterator iter{Iter{buffer}, PlusOneMutable{}};
      assert(iter[4] == 5);
      assert(buffer[4] == 4);

      static_assert(!noexcept(iter[4]));
      static_assert(cuda::std::is_same_v<int, decltype(iter[4])>);
    }

    {
      cuda::transform_iterator iter{Iter{buffer}, PlusOneNoexcept{}};
      assert(iter[4] == 5);
      assert(buffer[4] == 4);

      static_assert(noexcept(iter[4]) == noexcept(*cuda::std::declval<Iter>()));
      static_assert(cuda::std::is_same_v<int, decltype(iter[4])>);
    }

    {
      cuda::transform_iterator iter{Iter{buffer}, Increment{}};
      assert(iter[4] == 5);
      assert(buffer[4] == 5);

      static_assert(!noexcept(iter[4]));
      static_assert(cuda::std::is_same_v<int&, decltype(iter[4])>);
    }

    {
      cuda::transform_iterator iter{Iter{buffer}, IncrementRvalueRef{}};
      assert(iter[4] == 6);
      assert(buffer[4] == 6);

      static_assert(!noexcept(iter[4]));
      static_assert(cuda::std::is_same_v<int&&, decltype(iter[4])>);
    }

    {
      cuda::transform_iterator iter{Iter{buffer}, PlusWithMutableMember{3}};
      assert(iter[4] == 9);
      assert(iter[4] == 10);
      assert(iter[4] == 11);
      assert(buffer[4] == 6);

      static_assert(noexcept(iter[4]) == noexcept(*cuda::std::declval<Iter>()));
      static_assert(cuda::std::is_same_v<int, decltype(iter[4])>);
    }
  }
  else
  {
    static_assert(!can_subscript<Iter>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

#if TEST_COMPILER(MSVC)
  if (!cuda::std::is_constant_evaluated()) // MSVC complains about a non constant expression in the assignment
#endif // TEST_COMPILER(MSVC)
  { // Ensure that we can assign through projections
    using pair    = cuda::std::pair<int, int>;
    pair buffer[] = {{0, -1}, {1, -1}, {2, -1}, {3, -1}, {4, -1}, {5, -1}, {6, -1}, {7, -1}};
    cuda::transform_iterator iter{buffer, &pair::second};
    assert(iter[4] == -1);
    iter[4] = 42;
    assert(iter[4] == 42);

    const pair expected{4, 42};
    assert(buffer[4] == expected);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
