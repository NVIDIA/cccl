//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_iterator::operator{++,--,+=,-=}

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter>
_CCCL_CONCEPT can_decrement = _CCCL_REQUIRES_EXPR((Iter), Iter i)((--i));
template <class Iter>
_CCCL_CONCEPT can_post_decrement = _CCCL_REQUIRES_EXPR((Iter), Iter i)((i--));

template <class Iter>
_CCCL_CONCEPT can_plus_equal = _CCCL_REQUIRES_EXPR((Iter), Iter i)((i += 1));
template <class Iter>
_CCCL_CONCEPT can_minus_equal = _CCCL_REQUIRES_EXPR((Iter), Iter i)((i -= 1));

template <class Iter>
__host__ __device__ constexpr void test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  cuda::transform_iterator iter{Iter{buffer}, PlusOne{}};
  assert((++iter).base() == Iter{buffer + 1});

  if constexpr (cuda::std::forward_iterator<Iter>)
  {
    assert((iter++).base() == Iter{buffer + 1});
  }
  else
  {
    iter++;
    static_assert(cuda::std::is_same_v<decltype(iter++), void>);
  }
  assert(iter.base() == Iter{buffer + 2});

  if constexpr (cuda::std::bidirectional_iterator<Iter>)
  {
    assert((--iter).base() == Iter{buffer + 1});
    assert((iter--).base() == Iter{buffer + 1});
    assert(iter.base() == Iter{buffer});
  }
  else
  {
    static_assert(!can_decrement<Iter>);
    static_assert(!can_post_decrement<Iter>);
  }

  if constexpr (cuda::std::random_access_iterator<Iter>)
  {
    assert((iter += 4).base() == Iter{buffer + 4});
    assert((iter -= 3).base() == Iter{buffer + 1});
  }
  else
  {
    static_assert(!can_plus_equal<Iter>);
    static_assert(!can_minus_equal<Iter>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<cpp17_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
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
