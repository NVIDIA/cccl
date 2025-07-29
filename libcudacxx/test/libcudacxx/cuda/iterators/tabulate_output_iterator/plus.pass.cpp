//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iterator operator+(iterator i, difference_type n);
// friend constexpr iterator operator+(difference_type n, iterator i);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  basic_functor func{};

  cuda::tabulate_output_iterator iter1{func, 0};
  cuda::tabulate_output_iterator iter2{func, 0};
  assert(iter1 == iter2);
  assert(iter1 + 0 == iter1);
  assert(iter1 + 5 != iter2);
  assert(iter1 + 5 == cuda::std::ranges::next(iter2, 5));
  *iter1 = 0;

  static_assert(noexcept(iter2 + 5));
  static_assert(cuda::std::is_same_v<decltype(iter2 + (-5)), decltype(iter2)>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
