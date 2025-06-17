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
  basic_functor func{};

  { // <iterator> - difference_type
    cuda::tabulate_output_iterator iter1{func, 10};
    cuda::tabulate_output_iterator iter2{func, 10};
    assert(iter1 == iter2);
    assert(iter1 - 0 == iter2);
    assert(iter1 - 5 != iter2);
    *(iter1 - 5) = 5;

    static_assert(noexcept(iter2 - 5));
    static_assert(!cuda::std::is_reference_v<decltype(iter2 - 5)>);
  }

  { // <iterator> - <iterator>
    cuda::tabulate_output_iterator iter1{func, 5};
    cuda::tabulate_output_iterator iter2{func, 10};
    assert(iter1 - iter2 == 5);
    assert(iter1 - iter1 == 0);
    assert(iter2 - iter1 == -5);

    static_assert(noexcept(iter1 - iter2));
    static_assert(cuda::std::same_as<decltype(iter1 - iter2), int>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
