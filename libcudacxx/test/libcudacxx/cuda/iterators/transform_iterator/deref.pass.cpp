//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_iterator::operator*

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
__host__ __device__ constexpr void test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    cuda::transform_iterator iter{Iter{buffer}, PlusOne{}};
    assert(*iter == 1);
    assert(*cuda::std::as_const(iter) == 1);
    assert(*buffer == 0);

    static_assert(!noexcept(*iter));
    static_assert(cuda::std::is_same_v<int, decltype(*iter)>);
  }

  {
    cuda::transform_iterator iter{Iter{buffer}, PlusOneMutable{}};
    assert(*iter == 1);
    assert(*buffer == 0);

    static_assert(!noexcept(*iter));
    static_assert(cuda::std::is_same_v<int, decltype(*iter)>);
  }

  {
    cuda::transform_iterator iter{Iter{buffer}, PlusOneNoexcept{}};
    assert(*iter == 1);
    assert(*buffer == 0);

    static_assert(noexcept(*iter) == noexcept(*cuda::std::declval<Iter>()));
    static_assert(cuda::std::is_same_v<int, decltype(*iter)>);
  }

  {
    cuda::transform_iterator iter{Iter{buffer}, Increment{}};
    assert(*iter == 1);
    assert(*buffer == 1);

    static_assert(!noexcept(*iter));
    static_assert(cuda::std::is_same_v<int&, decltype(*iter)>);
  }

  {
    cuda::transform_iterator iter{Iter{buffer}, IncrementRvalueRef{}};
    assert(*iter == 2);
    assert(*buffer == 2);

    static_assert(!noexcept(*iter));
    static_assert(cuda::std::is_same_v<int&&, decltype(*iter)>);
  }

  {
    cuda::transform_iterator iter{Iter{buffer}, PlusWithMutableMember{3}};
    assert(*iter == 5);
    assert(*iter == 6);
    assert(*iter == 7);
    assert(*buffer == 2);

    static_assert(noexcept(*iter) == noexcept(*cuda::std::declval<Iter>()));
    static_assert(cuda::std::is_same_v<int, decltype(*iter)>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
