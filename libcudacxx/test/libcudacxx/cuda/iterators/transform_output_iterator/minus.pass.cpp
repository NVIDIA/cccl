//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iterator operator-(iterator i, difference_type n);
// friend constexpr difference_type operator-(const iterator& x, const iterator& y);

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PlusOne func{};

  { // <iterator> - difference_type
    cuda::transform_output_iterator iter1{buffer + 6, func};
    cuda::transform_output_iterator iter2{buffer + 6, func};
    assert(iter1 == iter2);
    assert(iter1 - 0 == iter2);
    assert(iter1 - 5 != iter2);
    assert((iter1 - 5).base() == buffer + 1);

    static_assert(noexcept(iter2 - 5));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
  }

  { // <iterator> - <iterator>
    cuda::transform_output_iterator iter1{buffer + 6, func};
    cuda::transform_output_iterator iter2{buffer + 3, func};
    assert(iter1 - iter2 == 3);
    assert(iter1 - iter1 == 0);
    assert(iter2 - iter1 == -3);

    static_assert(noexcept(iter1 - iter2));
    static_assert(cuda::std::same_as<decltype(iter1 - iter2), cuda::std::iter_difference_t<int*>>);
  }

  { // <iterator> - <iterator> not random access
    cuda::transform_output_iterator iter1{forward_sized_iterator{buffer + 6}, func};
    cuda::transform_output_iterator iter2{forward_sized_iterator{buffer + 3}, func};
    assert(iter1 - iter2 == 3);
    assert(iter1 - iter1 == 0);
    assert(iter2 - iter1 == -3);

    static_assert(noexcept(iter1 - iter2));
    static_assert(
      cuda::std::same_as<decltype(iter1 - iter2), cuda::std::iter_difference_t<forward_sized_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
